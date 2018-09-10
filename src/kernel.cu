#define GLM_FORCE_CUDA
#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <glm/glm.hpp>
#include "utilityCore.hpp"
#include "kernel.h"

#ifndef imax
#define imax( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef imin
#define imin( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

/**
* Check for CUDA errors; print and exit if there was a problem.
*/
void checkCUDAError(const char *msg, int line = -1) {
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    if (line >= 0) {
      fprintf(stderr, "Line %d: ", line);
    }
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}


/*****************
* Configuration *
*****************/

/*! Block size used for CUDA kernel launch. */
#define blockSize 128

// Parameters for the boids algorithm.
#define rule1Distance 5.0f
#define rule2Distance 3.0f
#define rule3Distance 5.0f

#define rule1Scale 0.01f
#define rule2Scale 0.1f
#define rule3Scale 0.1f

#define maxSpeed 1.0f

/*! Size of the starting area in simulation space. */
#define scene_scale 100.0f

/***********************************************
* Kernel state (pointers are device pointers) *
***********************************************/

int numObjects;
dim3 threadsPerBlock(blockSize);

// Buffers for the boid information.
glm::vec3 *dev_pos;
glm::vec3 *dev_vel1;
glm::vec3 *dev_vel2;

// Ping-pong buffer for shuffling when coherent
glm::vec3 *dev_shuffle_pos;
glm::vec3 *dev_shuffle_vel1;
glm::vec3 *dev_shuffle_vel2;

// For efficient sorting and the uniform grid. These should always be parallel.
int *dev_particleArrayIndices; // What index in dev_pos and dev_velX represents this particle?
int *dev_particleGridIndices; // What grid cell is this particle in?

// used in key-value pair sort
thrust::device_ptr<int> dev_thrust_particleArrayIndices;
thrust::device_ptr<int> dev_thrust_particleGridIndices;

int *dev_gridCellStartIndices; // What part of dev_particleArrayIndices belongs
int *dev_gridCellEndIndices;   // to this cell?

// These are automatically computed in Boids::initSimulation
int gridCellCount;
int gridSideCount;
float gridCellWidth;
float gridInverseCellWidth;
glm::vec3 gridMinimum;

/******************
* initSimulation *
******************/

__host__ __device__ unsigned int hash(unsigned int a) {
  a = (a + 0x7ed55d16) + (a << 12);
  a = (a ^ 0xc761c23c) ^ (a >> 19);
  a = (a + 0x165667b1) + (a << 5);
  a = (a + 0xd3a2646c) ^ (a << 9);
  a = (a + 0xfd7046c5) + (a << 3);
  a = (a ^ 0xb55a4f09) ^ (a >> 16);
  return a;
}

/**
* Function for generating a random vec3.
*/
__host__ __device__ glm::vec3 generateRandomVec3(float time, int index) {
  thrust::default_random_engine rng(hash((int)(index * time)));
  thrust::uniform_real_distribution<float> unitDistrib(-1, 1);

  return glm::vec3((float)unitDistrib(rng), (float)unitDistrib(rng), (float)unitDistrib(rng));
}

/**
* CUDA kernel for generating boids with a specified mass randomly around the star.
*/
__global__ void kernGenerateRandomPosArray(int time, int N, glm::vec3 * arr, float scale) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < N) {
    glm::vec3 rand = generateRandomVec3(time, index);
    arr[index].x = scale * rand.x;
    arr[index].y = scale * rand.y;
    arr[index].z = scale * rand.z;
  }
}

/**
* Initialize memory, update some globals
*/
void Boids::initSimulation(int N) {
  numObjects = N;
  dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

  // Malloc space for all buffers
  cudaMalloc((void**)&dev_pos, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_pos failed!");

  cudaMalloc((void**)&dev_vel1, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel1 failed!");

  cudaMalloc((void**)&dev_vel2, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel2 failed!");

  cudaMalloc((void**)&dev_shuffle_pos, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_shuffle_pos failed!");

  cudaMalloc((void**)&dev_shuffle_vel1, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_shuffle_vel1 failed!");

  cudaMalloc((void**)&dev_shuffle_vel2, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_shuffle_vel2 failed!");

  // Generate random starting position for the boids
  kernGenerateRandomPosArray<<<fullBlocksPerGrid, blockSize>>>(1, numObjects,
    dev_pos, scene_scale);
  checkCUDAErrorWithLine("kernGenerateRandomPosArray failed!");

  // computing grid params
  gridCellWidth = 2.0f * std::max(std::max(rule1Distance, rule2Distance), rule3Distance);
  int halfSideCount = (int)(scene_scale / gridCellWidth) + 1;
  gridSideCount = 2 * halfSideCount;

  gridCellCount = gridSideCount * gridSideCount * gridSideCount;
  gridInverseCellWidth = 1.0f / gridCellWidth;
  float halfGridWidth = gridCellWidth * halfSideCount;
  gridMinimum.x -= halfGridWidth;
  gridMinimum.y -= halfGridWidth;
  gridMinimum.z -= halfGridWidth;

  // Additional buffers for the uniform grid and coherent grid methods
  cudaMalloc((void**)&dev_particleArrayIndices, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_particleArrayIndices failed!");

  cudaMalloc((void**)&dev_particleGridIndices, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_particleGridIndices failed!");

  cudaMalloc((void**)&dev_gridCellStartIndices, gridCellCount * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_gridCellStartIndices failed!");

  cudaMalloc((void**)&dev_gridCellEndIndices, gridCellCount * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_gridCellEndIndices failed!");

  dev_thrust_particleArrayIndices = thrust::device_ptr<int>(dev_particleArrayIndices);
  dev_thrust_particleGridIndices = thrust::device_ptr<int>(dev_particleGridIndices);

  cudaDeviceSynchronize();
}


/******************
* copyBoidsToVBO *
******************/

/**
* Copy the boid positions into the VBO so that they can be drawn by OpenGL.
*/
__global__ void kernCopyPositionsToVBO(int N, glm::vec3 *pos, float *vbo, float s_scale) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);

  float c_scale = -1.0f / s_scale;

  if (index < N) {
    vbo[4 * index + 0] = pos[index].x * c_scale;
    vbo[4 * index + 1] = pos[index].y * c_scale;
    vbo[4 * index + 2] = pos[index].z * c_scale;
    vbo[4 * index + 3] = 1.0f;
  }
}

__global__ void kernCopyVelocitiesToVBO(int N, glm::vec3 *vel, float *vbo, float s_scale) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);

  if (index < N) {
    vbo[4 * index + 0] = vel[index].x + 0.3f;
    vbo[4 * index + 1] = vel[index].y + 0.3f;
    vbo[4 * index + 2] = vel[index].z + 0.3f;
    vbo[4 * index + 3] = 1.0f;
  }
}

/**
* Wrapper for call to the kernCopyboidsToVBO CUDA kernel.
*/
void Boids::copyBoidsToVBO(float *vbodptr_positions, float *vbodptr_velocities) {
  dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);

  kernCopyPositionsToVBO << <fullBlocksPerGrid, blockSize >> >(numObjects, dev_pos, vbodptr_positions, scene_scale);
  kernCopyVelocitiesToVBO << <fullBlocksPerGrid, blockSize >> >(numObjects, dev_vel1, vbodptr_velocities, scene_scale);

  checkCUDAErrorWithLine("copyBoidsToVBO failed!");

  cudaDeviceSynchronize();
}


/******************
* stepSimulation *
******************/

/**
* Compute the new velocity on the body with index `iSelf` due to the `N` boids
* in the `pos` and `vel` arrays.
*/
__device__ glm::vec3 computeVelocityChange(int N, int iSelf, const glm::vec3 *pos, const glm::vec3 *vel) {

  // Initialize the vectors that will be edited by the three rules
  glm::vec3 rule1_vel(0.0f, 0.0f, 0.0f);
  glm::vec3 rule2_vel(0.0f, 0.0f, 0.0f);
  glm::vec3 rule3_vel(0.0f, 0.0f, 0.0f);

  // Initialize other state relating to the rules
  glm::vec3 rule1_perceived_center(0.0f, 0.0f, 0.0f);
  glm::vec3 rule2_c_vel(0.0f, 0.0f, 0.0f);
  glm::vec3 rule3_perceived_vel(0.0f, 0.0f, 0.0f);
  
  // Loop through all of the boids
  int rule1Count = 0;
  int rule3Count = 0;
  for (int i = 0; i < N; ++i)
  {
    if (i != iSelf)
    {
      glm::vec3 difference = pos[i] - pos[iSelf];
      float len = glm::length(difference);

      // Rule 1: boids fly towards their local perceived center of mass, which excludes themselves
      if (len < rule1Distance)
      {
        rule1Count++;
        rule1_perceived_center += pos[i];
      }
      
      // Rule 2: boids try to stay a distance d away from each other
      if (len < rule2Distance)
      {
        rule2_c_vel -= (pos[i] - pos[iSelf]);
      }

      // Rule 3: boids try to match the speed of surrounding boids
      if (len < rule3Distance)
      {
        rule3Count++;
        if (glm::length(vel[i]) > 0)
        {
          rule3_perceived_vel += vel[i];
        }
      }
    }
  }

  // Scale the perceived center by the number of Boids
  if (rule1Count > 0)
  {
    rule1_perceived_center /= rule1Count;
    rule1_vel = (rule1_perceived_center - pos[iSelf]) * rule1Scale;
  }

  // Scale the percieved velocity of the flock by the number of Boids
  if (rule3Count > 0)
  {
    rule3_perceived_vel /= rule3Count;
    rule3_vel = rule3_perceived_vel * rule3Scale;
  }

  // Calculate the velocities from the rules
  rule2_vel = rule2_c_vel * rule2Scale;

  return vel[iSelf] + rule1_vel + rule2_vel + rule3_vel;
}

/**
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdateVelocityBruteForce(int N, glm::vec3 *pos,
  glm::vec3 *vel1, glm::vec3 *vel2) {
  
  // Get my index
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if (index >= N) {
    return;
  }

  // Compute my velocity
  glm::vec3 newVelocity = computeVelocityChange(N, index, pos, vel1);

  // Clamp the speed
  if (glm::length(newVelocity) > maxSpeed)
  {
    newVelocity = (newVelocity / glm::length(newVelocity)) * maxSpeed;
  }

  // Record the new velocity into vel2
  vel2[index] = newVelocity;
}

/**
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdatePos(int N, float dt, glm::vec3 *pos, glm::vec3 *vel) {
  // Update position by velocity
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if (index >= N) {
    return;
  }
  glm::vec3 thisPos = pos[index];
  thisPos += vel[index] * dt;

  // Wrap the boids around so we don't lose them
  thisPos.x = thisPos.x < -scene_scale ? scene_scale : thisPos.x;
  thisPos.y = thisPos.y < -scene_scale ? scene_scale : thisPos.y;
  thisPos.z = thisPos.z < -scene_scale ? scene_scale : thisPos.z;

  thisPos.x = thisPos.x > scene_scale ? -scene_scale : thisPos.x;
  thisPos.y = thisPos.y > scene_scale ? -scene_scale : thisPos.y;
  thisPos.z = thisPos.z > scene_scale ? -scene_scale : thisPos.z;

  pos[index] = thisPos;
}

__device__ int gridIndex3Dto1D(int x, int y, int z, int gridResolution) {
  return x + y * gridResolution + z * gridResolution * gridResolution;
}

__global__ void kernComputeIndices(int N, int gridResolution,
  glm::vec3 gridMin, float inverseCellWidth,
  glm::vec3 *pos, int *indices, int *gridIndices) {

  // get my index
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if (index >= N) {
    return;
  }
  
  // get my position relative to the minimum corner of the grid
  glm::vec3 myPos = pos[index];
  glm::vec3 gridLocation = myPos - gridMin;
  float myPosX = gridLocation.x;
  float myPosY = gridLocation.y;
  float myPosZ = gridLocation.z;
  
  // get my grid cell in 1D index
  int gridCellX = static_cast<int>(myPosX * inverseCellWidth);
  int gridCellY = static_cast<int>(myPosY * inverseCellWidth);
  int gridCellZ = static_cast<int>(myPosZ * inverseCellWidth);
  int gridIndex = gridIndex3Dto1D(gridCellX, gridCellY, gridCellZ, gridResolution);

  // Label me with the index of my grid cell.
  gridIndices[index] = gridIndex;

  // Set up a parallel array of integer indices as pointers to the actual boid data in pos and vel1/vel2
  indices[index] = index;
}

__global__ void kernResetIntBuffer(int N, int *intBuffer, int value) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < N) {
    intBuffer[index] = value;
  }
}

__global__ void kernIdentifyCellStartEnd(int N, int *particleGridIndices,
  int *gridCellStartIndices, int *gridCellEndIndices) {

  // get my index
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if (index >= N) {
    return;
  }

  // now just check that index, and the indices to the left and right
  int myGridIndex = particleGridIndices[index];

  // if the index to the left is different, or if there isn't an index to the left
  // then mark this as the start of grid #myGridIndex
  if (index == 0 || (index > 0 && myGridIndex != particleGridIndices[index - 1]))
  {
    gridCellStartIndices[myGridIndex] = index;
  }

  // if the index to the right is different, or if there isn't an index to the right 
  // then mark this as the end of grid #myGridIndex
  if (index == (N-1) || (index < (N - 1) && myGridIndex != particleGridIndices[index + 1]))
  {
    gridCellEndIndices[myGridIndex] = index;
  }
}

/*
 * Update a boid's velocity using the uniform grid to reduce the number of boids that need to be checked.
 */
__global__ void kernUpdateVelNeighborSearchScattered(
  int N, int gridResolution, glm::vec3 gridMin,
  float inverseCellWidth, float cellWidth,
  int *gridCellStartIndices, int *gridCellEndIndices,
  int *particleArrayIndices,
  glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2) {
 
  // first, as per usual, get my index
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if (index >= N) {
    return;
  }

  // Initialize the vectors that will be edited by the three rules
  glm::vec3 rule1_vel(0.0f, 0.0f, 0.0f);
  glm::vec3 rule2_vel(0.0f, 0.0f, 0.0f);
  glm::vec3 rule3_vel(0.0f, 0.0f, 0.0f);

  // Initialize other state relating to the rules
  glm::vec3 rule1_perceived_center(0.0f, 0.0f, 0.0f);
  glm::vec3 rule2_c_vel(0.0f, 0.0f, 0.0f);
  glm::vec3 rule3_perceived_vel(0.0f, 0.0f, 0.0f);
  
  /*** Now we'll get the grid cell and the octant within that cell that this particle is in ***/

  // get my position relative to the minimum corner of the grid
  glm::vec3 myPos = pos[particleArrayIndices[index]];
  glm::vec3 gridLocation = myPos - gridMin;
  float myPosX = gridLocation.x;
  float myPosY = gridLocation.y;
  float myPosZ = gridLocation.z;
  
  // get my grid cell in 3D float index
  float gridCellX = myPosX * inverseCellWidth;
  float gridCellY = myPosY * inverseCellWidth;
  float gridCellZ = myPosZ * inverseCellWidth;

  // also get my grid cell in int index
  int gridCellXIndex = static_cast<int>(gridCellX);
  int gridCellYIndex = static_cast<int>(gridCellY);
  int gridCellZIndex = static_cast<int>(gridCellZ);

  // also get the middle of each one of those grid cells
  float gridCellXMiddle = static_cast<float>(static_cast<int>(gridCellX)) + 0.5f;
  float gridCellYMiddle = static_cast<float>(static_cast<int>(gridCellY)) + 0.5f;
  float gridCellZMiddle = static_cast<float>(static_cast<int>(gridCellZ)) + 0.5f;
  
  // now determine which octant we are in by these three numbers
  int xSign = (gridCellX < gridCellXMiddle) ? -1 : 1;
  int ySign = (gridCellY < gridCellYMiddle) ? -1 : 1;
  int zSign = (gridCellZ < gridCellZMiddle) ? -1 : 1;

  /*** Now that we know our octant, we can check the neighbors in the neighboring cells ***/

  // for loop over the 8 neighbors to check for neighboring boids
  int rule1Count = 0;
  int rule3Count = 0;
  for (int i = 0; i < 8; ++i)
  {
    /*** first we have to figure out the 3D grid index of the neighbor we are looping over ***/

    // initialize neighbor's index to our index
    int neighborGridX = gridCellXIndex;
    int neighborGridY = gridCellYIndex;
    int neighborGridZ = gridCellZIndex;

    // based on the three numbers from earlier, calculate the true neighbor grid index
    if (i % 2 > 0)
    {
      neighborGridX += xSign;
    }

    if (i % 4 > 1)
    {
      neighborGridY += ySign;
    }
    
    if (i % 8 > 3)
    {
      neighborGridZ += zSign;
    }

    // if the neighbor's grid index is invalid, then skip it
    if (neighborGridX < 0 || neighborGridX > (gridResolution - 1) ||
      neighborGridY < 0 || neighborGridY > (gridResolution - 1) ||
      neighborGridZ < 0 || neighborGridZ > (gridResolution - 1))
    {
      continue;
    }
    
    // now that we have the neighbor's index, get that in 1D and find out where it starts and ends in the buffers
    int neighborGridIndex = gridIndex3Dto1D(neighborGridX, neighborGridY, neighborGridZ, gridResolution);
    int neighborGridParticleStart = gridCellStartIndices[neighborGridIndex];
    int neighborGridParticleEnd = gridCellEndIndices[neighborGridIndex];

    // if that grid cell doesn't have any particles in it or if there is some error then that's fine, just skip it
    if (neighborGridParticleStart == -1 || neighborGridParticleEnd == -1 || neighborGridParticleStart > neighborGridParticleEnd)
    {
      continue;
    }
 
    // for each particle within that neighboring grid cell
    for (int j = neighborGridParticleStart; j <= neighborGridParticleEnd; ++j)
    {
      // make sure it's not us
      if (j != index)
      {

        /*** apply the three rules ***/

        glm::vec3 difference = pos[particleArrayIndices[j]] - pos[particleArrayIndices[index]];
        float len = glm::length(difference);
     
        // Rule 1: boids fly towards their local perceived center of mass, which excludes themselves
        if (len < rule1Distance)
        {
          rule1Count++;
          rule1_perceived_center += pos[particleArrayIndices[j]];
        }

        // Rule 2: boids try to stay a distance d away from each other
        if (len < rule2Distance)
        {
          rule2_c_vel -= (pos[particleArrayIndices[j]] - pos[particleArrayIndices[index]]);
        }

        // Rule 3: boids try to match the speed of surrounding boids
        if (len < rule3Distance)
        {
          rule3Count++;
          if (glm::length(vel1[particleArrayIndices[j]]) > 0)
          {
            rule3_perceived_vel += vel1[particleArrayIndices[j]];
          }
        }
      }
    }
  }

  // Scale rules 1 and 3 by the number of Neighbors
  if (rule1Count > 0)
  {
    rule1_perceived_center /= rule1Count;
    rule1_vel = (rule1_perceived_center - pos[particleArrayIndices[index]]) * rule1Scale;
  }

  if (rule3Count > 0)
  {
    rule3_perceived_vel /= rule3Count;
    rule3_vel = rule3_perceived_vel * rule3Scale;
  }

  // Rule 2 doesn't need to be scaled
  rule2_vel = rule2_c_vel * rule2Scale;

  // calculate the new velocity by adding up all the rule results
  glm::vec3 newVelocity = vel1[particleArrayIndices[index]] + rule1_vel + rule2_vel + rule3_vel;
  
  // scale the velocity to account for max speed
  if (glm::length(newVelocity) > maxSpeed)
  {
    newVelocity = (newVelocity / glm::length(newVelocity)) * maxSpeed;
  }

  // put it into vel2 so we can later ping-pong it
  vel2[particleArrayIndices[index]] = newVelocity;
}

__global__ void kernUpdateVelNeighborSearchCoherent(
  int N, int gridResolution, glm::vec3 gridMin,
  float inverseCellWidth, float cellWidth,
  int *gridCellStartIndices, int *gridCellEndIndices,
  glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2) {

  // first, as per usual, get my index
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if (index >= N) {
    return;
  }

  // Initialize the vectors that will be edited by the three rules
  glm::vec3 rule1_vel(0.0f, 0.0f, 0.0f);
  glm::vec3 rule2_vel(0.0f, 0.0f, 0.0f);
  glm::vec3 rule3_vel(0.0f, 0.0f, 0.0f);

  // Initialize other state relating to the rules
  glm::vec3 rule1_perceived_center(0.0f, 0.0f, 0.0f);
  glm::vec3 rule2_c_vel(0.0f, 0.0f, 0.0f);
  glm::vec3 rule3_perceived_vel(0.0f, 0.0f, 0.0f);

  /*** Now we'll get the grid cell and the octant within that cell that this particle is in ***/
  
  // get my position relative to the minimum corner of the grid
  glm::vec3 myPos = pos[index];
  glm::vec3 gridLocation = myPos - gridMin;
  float myPosX = gridLocation.x;
  float myPosY = gridLocation.y;
  float myPosZ = gridLocation.z;
  
  // get my grid cell in 3D float index
  float gridCellX = myPosX * inverseCellWidth;
  float gridCellY = myPosY * inverseCellWidth;
  float gridCellZ = myPosZ * inverseCellWidth;

  // also get my grid cell in int index
  int gridCellXIndex = static_cast<int>(gridCellX);
  int gridCellYIndex = static_cast<int>(gridCellY);
  int gridCellZIndex = static_cast<int>(gridCellZ);

  // also get the middle of each one of those grid cells
  float gridCellXMiddle = static_cast<float>(static_cast<int>(gridCellX)) + 0.5f;
  float gridCellYMiddle = static_cast<float>(static_cast<int>(gridCellY)) + 0.5f;
  float gridCellZMiddle = static_cast<float>(static_cast<int>(gridCellZ)) + 0.5f;
  
  // now determine which octant we are in by the three numbers
  int xSign = (gridCellX < gridCellXMiddle) ? -1 : 1;
  int ySign = (gridCellY < gridCellYMiddle) ? -1 : 1;
  int zSign = (gridCellZ < gridCellZMiddle) ? -1 : 1;

  /*** Now that we know our octant, we can check the neighbors in the neighboring cells ***/

  // for loop over the 8 neighbors to check for neighboring boids
  int rule1Count = 0;
  int rule3Count = 0;
  for (int i = 0; i < 8; ++i)
  {
    /*** first we have to figure out the 3D grid index of the neighbor we are looping over ***/

    // initialize neighbor's index to our index
    int neighborGridX = gridCellXIndex;
    int neighborGridY = gridCellYIndex;
    int neighborGridZ = gridCellZIndex;

    // based on the three numbers from earlier, calculate the true neighbor grid index
    if (i % 2 > 0)
    {
      neighborGridX += xSign;
    }

    if (i % 4 > 1)
    {
      neighborGridY += ySign;
    }
    
    if (i % 8 > 3)
    {
      neighborGridZ += zSign;
    }

    // if the neighbor's grid index is invalid, then skip it
    if (neighborGridX < 0 || neighborGridX > (gridResolution - 1) ||
      neighborGridY < 0 || neighborGridY > (gridResolution - 1) ||
      neighborGridZ < 0 || neighborGridZ > (gridResolution - 1))
    {
      continue;
    }
    
    // now that we have the neighbor's index, get that in 1D and find out where it starts and ends in the buffers
    int neighborGridIndex = gridIndex3Dto1D(neighborGridX, neighborGridY, neighborGridZ, gridResolution);
    int neighborGridParticleStart = gridCellStartIndices[neighborGridIndex];
    int neighborGridParticleEnd = gridCellEndIndices[neighborGridIndex];

    // if that grid cell doesn't have any particles in it or if there is some error then that's fine, just skip it
    if (neighborGridParticleStart == -1 || neighborGridParticleEnd == -1 || neighborGridParticleStart > neighborGridParticleEnd)
    {
      continue;
    }
    
    // for each particle within that neighboring grid cell
    for (int j = neighborGridParticleStart; j <= neighborGridParticleEnd; ++j)
    {
      // make sure it's not us
      if (j != index)
      {

        /*** apply the three rules ***/

        glm::vec3 difference = pos[j] - pos[index];
        float len = glm::length(difference);
     
        // Rule 1: boids fly towards their local perceived center of mass, which excludes themselves
        if (len < rule1Distance)
        {
          rule1Count++;
          rule1_perceived_center += pos[j];
        }

        // Rule 2: boids try to stay a distance d away from each other
        if (len < rule2Distance)
        {
          rule2_c_vel -= (pos[j] - pos[index]);
        }

        // Rule 3: boids try to match the speed of surrounding boids
        if (len < rule3Distance)
        {
          rule3Count++;
          if (glm::length(vel1[j]) > 0)
          {
            rule3_perceived_vel += vel1[j];
          }
        }
      }
    }
  }

  // Scale rules 1 and 3 by the number of Neighbors
  if (rule1Count > 0)
  {
    rule1_perceived_center /= rule1Count;
    rule1_vel = (rule1_perceived_center - pos[index]) * rule1Scale;
  }

  if (rule3Count > 0)
  {
    rule3_perceived_vel /= rule3Count;
    rule3_vel = rule3_perceived_vel * rule3Scale;
  }

  // Rule 2 doesn't need to be scaled
  rule2_vel = rule2_c_vel * rule2Scale;

  // calculate the new velocity by adding up all the rule results
  glm::vec3 newVelocity = vel1[index] + rule1_vel + rule2_vel + rule3_vel;
  
  // scale the velocity to account for max speed
  if (glm::length(newVelocity) > maxSpeed)
  {
    newVelocity = (newVelocity / glm::length(newVelocity)) * maxSpeed;
  }

  // put it into vel2 so we can ping-pong it later
  vel2[index] = newVelocity;
}

/*
 * Takes pos, vel1, vel2 and shuffles them based on the indices in particleArrayIndices
 * editing pos_shuffle, vel1_shuffle, and vel2_shuffle in the process
 */
__global__ void kernShuffle(int N, int* particleArrayIndices, glm::vec3* pos,
  glm::vec3* pos_shuffle, glm::vec3* vel1, glm::vec3* vel1_shuffle,
  glm::vec3* vel2, glm::vec3* vel2_shuffle) {

  // get the index of the thing to switch
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if (index >= N) {
    return;
  }

  // shuffle all of the indices in parallel
  pos_shuffle[index] = pos[particleArrayIndices[index]];
  vel1_shuffle[index] = vel1[particleArrayIndices[index]];
  vel2_shuffle[index] = vel2[particleArrayIndices[index]];
}

/**
* Step the entire N-body simulation by `dt` seconds.
*/
void Boids::stepSimulationNaive(float dt) {
  dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);

  // Brute force the velocity Naively
  kernUpdateVelocityBruteForce << <fullBlocksPerGrid, blockSize >> > (numObjects, dev_pos, dev_vel1, dev_vel2);

  // Update position
  kernUpdatePos << <fullBlocksPerGrid, blockSize >> > (numObjects, dt, dev_pos, dev_vel1);

  // ping-pong the velocity buffers
  glm::vec3 *temp = dev_vel1;
  dev_vel1 = dev_vel2;
  dev_vel2 = temp;
}

void Boids::stepSimulationScatteredGrid(float dt) {
  dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);

  // Assign all particles a grid index
  kernComputeIndices << <fullBlocksPerGrid, blockSize >> > (numObjects, gridSideCount, gridMinimum,
                                                            gridInverseCellWidth, dev_pos, dev_particleArrayIndices, 
                                                            dev_particleGridIndices);
  checkCUDAErrorWithLine("kernComputeIndices failed!");

  cudaDeviceSynchronize();

  // Unstable key sort using Thrust
  thrust::sort_by_key(dev_thrust_particleGridIndices, dev_thrust_particleGridIndices + (numObjects - 1), 
                      dev_thrust_particleArrayIndices);
  checkCUDAErrorWithLine("thrust sort_by_key failed!");

  // reset the grid start and end indices back to their defaults of -1 to denote empty grid
  kernResetIntBuffer << <fullBlocksPerGrid, blockSize >> > (numObjects, dev_gridCellStartIndices, -1);
  checkCUDAErrorWithLine("kernResetIntBuffer failed!");

  kernResetIntBuffer << <fullBlocksPerGrid, blockSize >> > (numObjects, dev_gridCellEndIndices, -1);
  checkCUDAErrorWithLine("kernResetIntBuffer failed!");

  cudaDeviceSynchronize();

  // find the start and end indices of each grid
  kernIdentifyCellStartEnd << <fullBlocksPerGrid, blockSize >> > (numObjects, dev_particleGridIndices,
                                                                  dev_gridCellStartIndices, dev_gridCellEndIndices);
  checkCUDAErrorWithLine("kernIdentifyCellStartEnd failed!");

  cudaDeviceSynchronize();  

  // Perform velocity updates using neighbor search
  kernUpdateVelNeighborSearchScattered << <fullBlocksPerGrid, blockSize >> > (numObjects, gridSideCount,
                                                                              gridMinimum, gridInverseCellWidth, gridCellWidth, 
                                                                              dev_gridCellStartIndices, dev_gridCellEndIndices, 
                                                                              dev_particleArrayIndices, dev_pos, 
                                                                              dev_vel1, dev_vel2);
  checkCUDAErrorWithLine("kernUpdateVelNeighborSearchScattered failed!");
 
  cudaDeviceSynchronize();

  // Update positions
  kernUpdatePos << <fullBlocksPerGrid, blockSize >> > (numObjects, dt, dev_pos, dev_vel1);
  checkCUDAErrorWithLine("kernUpdatePos failed!");

  // Ping-pong vel1 and vel2
  glm::vec3 *temp = dev_vel1;
  dev_vel1 = dev_vel2;
  dev_vel2 = temp;
}

void Boids::stepSimulationCoherentGrid(float dt) {
  dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);

  // Assign all particles a grid index
  kernComputeIndices << <fullBlocksPerGrid, blockSize >> > (numObjects, gridSideCount, gridMinimum,
                                                            gridInverseCellWidth, dev_pos, dev_particleArrayIndices, 
                                                            dev_particleGridIndices);
  checkCUDAErrorWithLine("kernComputeIndices failed!");

  cudaDeviceSynchronize();

  // Unstable key sort using Thrust
  thrust::sort_by_key(dev_thrust_particleGridIndices, dev_thrust_particleGridIndices + (numObjects - 1), 
                      dev_thrust_particleArrayIndices);
  checkCUDAErrorWithLine("thrust sort_by_key failed!");

  // reset the grid start and end indices back to their defaults of -1 to denote empty grid
  kernResetIntBuffer << <fullBlocksPerGrid, blockSize >> > (numObjects, dev_gridCellStartIndices, -1);
  checkCUDAErrorWithLine("kernResetIntBuffer failed!");

  kernResetIntBuffer << <fullBlocksPerGrid, blockSize >> > (numObjects, dev_gridCellEndIndices, -1);
  checkCUDAErrorWithLine("kernResetIntBuffer failed!");

  cudaDeviceSynchronize();

  // find the start and end indices of each grid
  kernIdentifyCellStartEnd << <fullBlocksPerGrid, blockSize >> > (numObjects, dev_particleGridIndices,
                                                                  dev_gridCellStartIndices, dev_gridCellEndIndices);
  checkCUDAErrorWithLine("kernIdentifyCellStartEnd failed!");

  cudaDeviceSynchronize();  
  
  // COHERENT ONLY: reshuffle all of the particle data
  kernShuffle << <fullBlocksPerGrid, blockSize >> > (numObjects, dev_particleArrayIndices, dev_pos, 
                                                    dev_shuffle_pos, dev_vel1, dev_shuffle_vel1, dev_vel2, dev_shuffle_vel2);
  checkCUDAErrorWithLine("kernShuffle failed!");

  cudaDeviceSynchronize();

  // Perform velocity updates using neighbor search
  kernUpdateVelNeighborSearchCoherent << <fullBlocksPerGrid, blockSize >> > (numObjects, gridSideCount, gridMinimum, 
                                                                             gridInverseCellWidth, gridCellWidth, 
                                                                             dev_gridCellStartIndices, dev_gridCellEndIndices,
                                                                             dev_shuffle_pos, dev_shuffle_vel1, dev_shuffle_vel2);
  checkCUDAErrorWithLine("kernUpdateVelNeighborSearchScattered failed!");
 
  cudaDeviceSynchronize();

  // Update positions
  kernUpdatePos << <fullBlocksPerGrid, blockSize >> > (numObjects, dt, dev_shuffle_pos, dev_shuffle_vel1);
  checkCUDAErrorWithLine("kernUpdatePos failed!");


  // ping-pong the newly shuffled buffers
  glm::vec3 *pingPongTemp1 = dev_pos;
  glm::vec3 *pingPongTemp2 = dev_vel1;
  glm::vec3 *pingPongTemp3 = dev_vel2;
  dev_pos = dev_shuffle_pos;
  dev_vel1 = dev_shuffle_vel1;
  dev_vel2 = dev_shuffle_vel2;
  dev_shuffle_pos = pingPongTemp1;
  dev_shuffle_vel1 = pingPongTemp2;
  dev_shuffle_vel2 = pingPongTemp3;

  // ping pong vel1 and vel2
  glm::vec3 *temp = dev_vel1;
  dev_vel1 = dev_vel2;
  dev_vel2 = temp;
}

void Boids::endSimulation() {
  // free main buffers
  cudaFree(dev_vel1);
  cudaFree(dev_vel2);
  cudaFree(dev_pos);

  // free ping-pong buffers
  cudaFree(dev_shuffle_vel1);
  cudaFree(dev_shuffle_vel2);
  cudaFree(dev_shuffle_pos);

  // free grid state buffers
  cudaFree(dev_particleArrayIndices);
  cudaFree(dev_particleGridIndices);
  cudaFree(dev_gridCellStartIndices);
  cudaFree(dev_gridCellEndIndices);
}

void Boids::unitTest() {
  // test unstable sort
  int *dev_intKeys;
  int *dev_intValues;
  int N = 10;

  std::unique_ptr<int[]>intKeys{ new int[N] };
  std::unique_ptr<int[]>intValues{ new int[N] };

  intKeys[0] = 0; intValues[0] = 0;
  intKeys[1] = 1; intValues[1] = 1;
  intKeys[2] = 0; intValues[2] = 2;
  intKeys[3] = 3; intValues[3] = 3;
  intKeys[4] = 0; intValues[4] = 4;
  intKeys[5] = 2; intValues[5] = 5;
  intKeys[6] = 2; intValues[6] = 6;
  intKeys[7] = 0; intValues[7] = 7;
  intKeys[8] = 5; intValues[8] = 8;
  intKeys[9] = 6; intValues[9] = 9;

  cudaMalloc((void**)&dev_intKeys, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_intKeys failed!");

  cudaMalloc((void**)&dev_intValues, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_intValues failed!");

  dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

  std::cout << "before unstable sort: " << std::endl;
  for (int i = 0; i < N; i++) {
    std::cout << "  key: " << intKeys[i];
    std::cout << " value: " << intValues[i] << std::endl;
  }

  // How to copy data to the GPU
  cudaMemcpy(dev_intKeys, intKeys.get(), sizeof(int) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_intValues, intValues.get(), sizeof(int) * N, cudaMemcpyHostToDevice);

  // Wrap device vectors in thrust iterators for use with thrust.
  thrust::device_ptr<int> dev_thrust_keys(dev_intKeys);
  thrust::device_ptr<int> dev_thrust_values(dev_intValues);
  thrust::sort_by_key(dev_thrust_keys, dev_thrust_keys + N, dev_thrust_values);

  // How to copy data back to the CPU side from the GPU
  cudaMemcpy(intKeys.get(), dev_intKeys, sizeof(int) * N, cudaMemcpyDeviceToHost);
  cudaMemcpy(intValues.get(), dev_intValues, sizeof(int) * N, cudaMemcpyDeviceToHost);
  checkCUDAErrorWithLine("memcpy back failed!");

  std::cout << "after unstable sort: " << std::endl;
  for (int i = 0; i < N; i++) {
    std::cout << "  key: " << intKeys[i];
    std::cout << " value: " << intValues[i] << std::endl;
  }

  // cleanup
  cudaFree(dev_intKeys);
  cudaFree(dev_intValues);
  checkCUDAErrorWithLine("cudaFree failed!");
  return;
}
