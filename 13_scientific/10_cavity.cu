#include <iostream>
#include <vector>
#include <fstream>
#include <cuda_runtime.h>
using namespace std;

#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16

__global__ void init_arrays_kernel(float* u, float* v, float* p, float* b, int nx, int ny) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < nx && j < ny) {
        u[j * nx + i] = 0.0f;
        v[j * nx + i] = 0.0f;
        p[j * nx + i] = 0.0f;
        b[j * nx + i] = 0.0f;
    }
}

// Compute b[j][i]
__global__ void compute_b_kernel(float* u, float* v, float* b, int nx, int ny, double dx, double dy, double dt, double rho) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i > 0 && i < nx - 1 && j > 0 && j < ny - 1) {
        float dudx = (u[j * nx + i + 1] - u[j * nx + i - 1]) / (2.0f * dx);
        float dvdy = (v[(j + 1) * nx + i] - v[(j - 1) * nx + i]) / (2.0f * dy);
        float dudy = (u[(j + 1) * nx + i] - u[(j - 1) * nx + i]) / (2.0f * dy);
        float dvdx = (v[j * nx + i + 1] - v[j * nx + i - 1]) / (2.0f * dx);

        float divergence = dudx + dvdy;
        float nonlinear = dudx * dudx + 2.0f * dudy * dvdx + dvdy * dvdy;

        b[j * nx + i] = rho * ((divergence / dt) - nonlinear);
    }
}

// Compute p[j][i]
__global__ void compute_p_kernel(float* p, float* pn, float* b, int nx, int ny, double dx, double dy) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i > 0 && i < nx - 1 && j > 0 && j < ny - 1) {
        float weighted_x = dy * dy * (pn[j * nx + i + 1] + pn[j * nx + i - 1]);
        float weighted_y = dx * dx * (pn[(j + 1) * nx + i] + pn[(j - 1) * nx + i]);
        float rhs_term = b[j * nx + i] * dx * dx * dy * dy;
        float denom_inv = 1.0f / (2.0f * (dx * dx + dy * dy));

        p[j * nx + i] = (weighted_x + weighted_y - rhs_term) * denom_inv;
    }
}

__global__ void update_p_boundary_kernel(float* p, int nx, int ny) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // handle left and right boundaries for p
    if (idx < ny) { 
        p[idx * nx + nx - 1] = p[idx * nx + nx - 2];
        p[idx * nx + 0] = p[idx * nx + 1];
    }
    // handle top and bottom boundaries for p
    if (idx < nx) { 
        p[0 * nx + idx] = p[1 * nx + idx];
        p[(ny - 1) * nx + idx] = 0.0f;
    }
}

__global__ void copy_uv_kernel(float* u, float* v, float* un, float* vn, int nx, int ny) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < nx && j < ny) {
        un[j * nx + i] = u[j * nx + i];
        vn[j * nx + i] = v[j * nx + i];
    }
}

// Compute u[j][i] and v[j][i]
__global__ void compute_uv_kernel(float* u, float* v, float* un, float* vn, float* p, int nx, int ny, double dx, double dy, double dt, double rho, double nu) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i > 0 && i < nx - 1 && j > 0 && j < ny - 1) {
        //----- update u[j][i] --------
        float adv_u_x = un[j * nx + i] * (un[j * nx + i] - un[j * nx + i - 1]) / dx;
        float adv_u_y = un[j * nx + i] * (un[j * nx + i] - un[(j - 1) * nx + i]) / dy;

        float gradp_u  = (p[j * nx + i + 1] - p[j * nx + i - 1]) / (2.0f * rho * dx);

        float lap_u_x  = (un[j * nx + i + 1] - 2.0f * un[j * nx + i] + un[j * nx + i - 1]) / (dx * dx);
        float lap_u_y  = (un[(j + 1) * nx + i] - 2.0f * un[j * nx + i] + un[(j - 1) * nx + i]) / (dy * dy);

        u[j * nx + i] = un[j * nx + i]
                        - dt * (adv_u_x + adv_u_y)
                        - dt * gradp_u
                        + nu * dt * (lap_u_x + lap_u_y);

        //----- update v[j][i] --------
        float adv_v_x = vn[j * nx + i] * (vn[j * nx + i] - vn[j * nx + i - 1]) / dx;
        float adv_v_y = vn[j * nx + i] * (vn[j * nx + i] - vn[(j - 1) * nx + i]) / dy;

        float gradp_v  = (p[(j + 1) * nx + i] - p[(j - 1) * nx + i]) / (2.0f * rho * dy);

        float lap_v_x  = (vn[j * nx + i + 1] - 2.0f * vn[j * nx + i] + vn[j * nx + i - 1]) / (dx * dx);
        float lap_v_y  = (vn[(j + 1) * nx + i] - 2.0f * vn[j * nx + i] + vn[(j - 1) * nx + i]) / (dy * dy);

        v[j * nx + i] = vn[j * nx + i]
                        - dt * (adv_v_x + adv_v_y)
                        - dt * gradp_v
                        + nu * dt * (lap_v_x + lap_v_y);
    }
}


__global__ void update_uv_boundary_kernel(float* u, float* v, int nx, int ny) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // handle left and right boundaries for u and v
    if (idx < ny) { 
        u[idx * nx + 0] = 0.0f;
        u[idx * nx + nx - 1] = 0.0f;
        v[idx * nx + 0] = 0.0f;
        v[idx * nx + nx - 1] = 0.0f;
    }
    // handle top and bottom boundaries for u and v
    if (idx < nx) { 
        u[0 * nx + idx] = 0.0f;
        u[(ny - 1) * nx + idx] = 1.0f; 
        v[0 * nx + idx] = 0.0f;
        v[(ny - 1) * nx + idx] = 0.0f;
    }
}


int main() {
    int nx = 41;
    int ny = 41;
    int nt = 500;
    int nit = 50;
    double dx = 2.0 / (nx - 1);
    double dy = 2.0 / (ny - 1);
    double dt = .01;
    double rho = 1.0;
    double nu = .02;

    size_t array_size = nx * ny * sizeof(float);

    // host arrays
    float *h_u, *h_v, *h_p, *h_b;
    float *h_un, *h_vn, *h_pn;

    // device arrays
    float *d_u, *d_v, *d_p, *d_b;
    float *d_un, *d_vn, *d_pn;

    h_u = (float*)malloc(array_size);
    h_v = (float*)malloc(array_size);
    h_p = (float*)malloc(array_size);
    h_b = (float*)malloc(array_size);
    h_un = (float*)malloc(array_size);
    h_vn = (float*)malloc(array_size);
    h_pn = (float*)malloc(array_size);

    cudaMalloc((void**)&d_u, array_size);
    cudaMalloc((void**)&d_v, array_size);
    cudaMalloc((void**)&d_p, array_size);
    cudaMalloc((void**)&d_b, array_size);
    cudaMalloc((void**)&d_un, array_size);
    cudaMalloc((void**)&d_vn, array_size);
    cudaMalloc((void**)&d_pn, array_size);

    // Initialize host arrays to 0
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            h_u[j * nx + i] = 0.0f;
            h_v[j * nx + i] = 0.0f;
            h_p[j * nx + i] = 0.0f;
            h_b[j * nx + i] = 0.0f;
        }
    }

    cudaMemcpy(d_u, h_u, array_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v, array_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_p, h_p, array_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, array_size, cudaMemcpyHostToDevice);

    dim3 dimGrid((nx + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X, (ny + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y);
    dim3 dimBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y);

    init_arrays_kernel<<<dimGrid, dimBlock>>>(d_u, d_v, d_p, d_b, nx, ny);
    cudaDeviceSynchronize();


    ofstream ufile("u.dat");
    ofstream vfile("v.dat");
    ofstream pfile("p.dat");

    for (int n = 0; n < nt; n++) {
        // compute b
        compute_b_kernel<<<dimGrid, dimBlock>>>(d_u, d_v, d_b, nx, ny, dx, dy, dt, rho);
        cudaDeviceSynchronize();

        for (int it = 0; it < nit; it++) {
            // copy p to pn
            copy_uv_kernel<<<dimGrid, dimBlock>>>(d_p, d_p, d_pn, d_pn, nx, ny); // Reusing copy kernel, p to pn
            cudaDeviceSynchronize();

            // compute p
            compute_p_kernel<<<dimGrid, dimBlock>>>(d_p, d_pn, d_b, nx, ny, dx, dy);
            cudaDeviceSynchronize();

            // update p boundary conditions
            update_p_boundary_kernel<<<dimGrid.x, dimBlock.x>>>(d_p, nx, ny); 
            cudaDeviceSynchronize();
        }

        copy_uv_kernel<<<dimGrid, dimBlock>>>(d_u, d_v, d_un, d_vn, nx, ny);
        cudaDeviceSynchronize();

        // compute u and v
        compute_uv_kernel<<<dimGrid, dimBlock>>>(d_u, d_v, d_un, d_vn, d_p, nx, ny, dx, dy, dt, rho, nu);
        cudaDeviceSynchronize();

        // update u and v boundary conditions
        update_uv_boundary_kernel<<<dimGrid.x, dimBlock.x>>>(d_u, d_v, nx, ny); 
        cudaDeviceSynchronize();

        if (n % 10 == 0) {
            cudaMemcpy(h_u, d_u, array_size, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_v, d_v, array_size, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_p, d_p, array_size, cudaMemcpyDeviceToHost);

            for (int j = 0; j < ny; j++)
                for (int i = 0; i < nx; i++)
                    ufile << h_u[j * nx + i] << " ";
            ufile << "\n";
            for (int j = 0; j < ny; j++)
                for (int i = 0; i < nx; i++)
                    vfile << h_v[j * nx + i] << " ";
            vfile << "\n";
            for (int j = 0; j < ny; j++)
                for (int i = 0; i < nx; i++)
                    pfile << h_p[j * nx + i] << " ";
            pfile << "\n";
        }
    }

    ufile.close();
    vfile.close();
    pfile.close();

    cudaFree(d_u);
    cudaFree(d_v);
    cudaFree(d_p);
    cudaFree(d_b);
    cudaFree(d_un);
    cudaFree(d_vn);
    cudaFree(d_pn);

    free(h_u);
    free(h_v);
    free(h_p);
    free(h_b);
    free(h_un);
    free(h_vn);
    free(h_pn);

    return 0;
}
