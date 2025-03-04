#include "Sets.cuh"

#include <device_launch_parameters.h>
#include <math.h>

namespace sets
{
	__device__ int mandelbrot_kernel(double cr, double ci)
	{
		int		n = 0;
		double	x = 0;
		double	y = 0;
		double	x2 = 0;
		double	y2 = 0;

		while (x2 + y2 <= 4.0 && ++n < MAX_ITER + 1)
		{
			y = 2 * x * y + ci;
			x = x2 - y2 + cr;
			x2 = x * x;
			y2 = y * y;
		}
		return (n - 1);
	}

	__device__ double box_fold_kernel(double v)
	{
		if (v > 1)
			v = 2 - v;
		else if (v < -1)
			v = -2 - v;

		return v;
	}

	__device__ double ball_fold_kernel(double r, double m)
	{
		if (m < r)
			m = m / (r * r);
		else if (m < 1)
			m = 1 / (m * m);

		return m;
	}

	__device__ int mandelbox_kernel(double fx, double sx, double rx, double cr, double ci)
	{
		int		n = 0;
		double	vr = cr;
		double	vi = ci;
		double	mag = 0;

		while (++n < MAX_ITER + 1)
		{
			vr = fx * box_fold_kernel(vr);
			vi = fx * box_fold_kernel(vi);

			mag = sqrt(vr * vr + vi * vi);

			vr = vr * sx * ball_fold_kernel(rx, mag) + cr;
			vi = vi * sx * ball_fold_kernel(rx, mag) + ci;

			if (sqrt(mag) > 2)
				break;
		}
		return (n - 1);
	}

	__device__ int julia_kernel(double ki, double kr, double zr, double zi)
	{
		int		n = 0;
		double	tmp;

		while (++n < MAX_ITER + 1)
		{
			if ((zi * zi + zr * zr) > 4.0)
				break;

			tmp = 2 * zr * zi + ki;
			zr = zr * zr - zi * zi + kr;
			zi = tmp;
		}
		return (n - 1);
	}

	__device__ int tricorn_kernel(double cr, double ci)
	{
		int		n = 0;
		double	zr = cr;
		double	zi = ci;
		double	tmp;

		while (++n < MAX_ITER + 1)
		{
			if ((zr * zr + zi * zi) > 4.0)
				break;

			tmp = -2 * zr * zi + ci;
			zr = zr * zr - zi * zi + cr;
			zi = tmp;
		}
		return (n - 1);
	}

	__global__ void calculate_fractal(
		int* output,
		double x_min, double y_min,
		double dx, double dy,
		int width, int height,
		SET set_type,
		double julia_kr, double julia_ki,
		double mandelbox_fx, double mandelbox_sx, double mandelbox_rx)
	{
		int x = blockIdx.x * blockDim.x + threadIdx.x;
		int y = blockIdx.y * blockDim.y + threadIdx.y;

		if (x >= width || y >= height)
			return;

		double cr = x_min + x * dx;
		double ci = y_min + y * dy;
		int iterations = 0;

		switch (set_type)
		{
		case SET::MANDELBROT:
			iterations = mandelbrot_kernel(cr, ci);
			break;
		case SET::MANDELBOX:
			iterations = mandelbox_kernel(mandelbox_fx, mandelbox_sx, mandelbox_rx, cr, ci);
			break;
		case SET::TRICORN:
			iterations = tricorn_kernel(cr, ci);
			break;
		case SET::JULIA:
			iterations = julia_kernel(julia_ki, julia_kr, cr, ci);
			break;
		}

		output[y * width + x] = iterations;
	}

	cudaError_t calculate_mandelbrot(
		int* output,
		double x_min, double x_max,
		double y_min, double y_max,
		int width, int height)
	{
		int* dev_output = nullptr;
		cudaError_t cudaStatus;

		double dx = (x_max - x_min) / width;
		double dy = (y_max - y_min) / height;

		cudaStatus = cudaMalloc((void**)&dev_output, width * height * sizeof(int));
		if (cudaStatus != cudaSuccess)
			return cudaStatus;

		dim3 blockDim(16, 16);
		dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

		calculate_fractal << <gridDim, blockDim >> > (
			dev_output,
			x_min, y_min,
			dx, dy,
			width, height,
			SET::MANDELBROT,
			0.0, 0.0,
			0.0, 0.0, 0.0
		);

		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess)
		{
			cudaFree(dev_output);
			return cudaStatus;
		}

		cudaStatus = cudaMemcpy(output, dev_output, width * height * sizeof(int), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess)
		{
			cudaFree(dev_output);
			return cudaStatus;
		}

		cudaFree(dev_output);
		return cudaSuccess;
	}

	cudaError_t calculate_mandelbox(
		int* output,
		double x_min, double x_max,
		double y_min, double y_max,
		double fx, double sx, double rx,
		int width, int height)
	{
		int* dev_output = nullptr;
		cudaError_t cudaStatus;

		double dx = (x_max - x_min) / width;
		double dy = (y_max - y_min) / height;

		cudaStatus = cudaMalloc((void**)&dev_output, width * height * sizeof(int));
		if (cudaStatus != cudaSuccess)
			return cudaStatus;

		dim3 blockDim(16, 16);
		dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

		calculate_fractal << <gridDim, blockDim >> > (
			dev_output,
			x_min, y_min,
			dx, dy,
			width, height,
			SET::MANDELBOX,
			0.0, 0.0,
			fx, sx, rx
		);

		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess)
		{
			cudaFree(dev_output);
			return cudaStatus;
		}

		cudaStatus = cudaMemcpy(output, dev_output, width * height * sizeof(int), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess)
		{
			cudaFree(dev_output);
			return cudaStatus;
		}

		cudaFree(dev_output);
		return cudaSuccess;
	}

	cudaError_t calculate_julia(
		int* output,
		double x_min, double x_max,
		double y_min, double y_max,
		double kr, double ki,
		int width, int height)
	{
		int* dev_output = nullptr;
		cudaError_t cudaStatus;

		double dx = (x_max - x_min) / width;
		double dy = (y_max - y_min) / height;

		cudaStatus = cudaMalloc((void**)&dev_output, width * height * sizeof(int));
		if (cudaStatus != cudaSuccess)
			return cudaStatus;

		dim3 blockDim(16, 16);
		dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

		calculate_fractal << <gridDim, blockDim >> > (
			dev_output,
			x_min, y_min,
			dx, dy,
			width, height,
			SET::JULIA,
			kr, ki,
			0.0, 0.0, 0.0
		);

		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess)
		{
			cudaFree(dev_output);
			return cudaStatus;
		}

		cudaStatus = cudaMemcpy(output, dev_output, width * height * sizeof(int), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess)
		{
			cudaFree(dev_output);
			return cudaStatus;
		}

		cudaFree(dev_output);
		return cudaSuccess;
	}

	cudaError_t calculate_tricorn(
		int* output,
		double x_min, double x_max,
		double y_min, double y_max,
		int width, int height)
	{
		int* dev_output = nullptr;
		cudaError_t cudaStatus;

		double dx = (x_max - x_min) / width;
		double dy = (y_max - y_min) / height;

		cudaStatus = cudaMalloc((void**)&dev_output, width * height * sizeof(int));
		if (cudaStatus != cudaSuccess)
			return cudaStatus;

		dim3 blockDim(16, 16);
		dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

		calculate_fractal << <gridDim, blockDim >> > (
			dev_output,
			x_min, y_min,
			dx, dy,
			width, height,
			SET::TRICORN,
			0.0, 0.0,
			0.0, 0.0, 0.0
		);

		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess)
		{
			cudaFree(dev_output);
			return cudaStatus;
		}

		cudaStatus = cudaMemcpy(output, dev_output, width * height * sizeof(int), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess)
		{
			cudaFree(dev_output);
			return cudaStatus;
		}

		cudaFree(dev_output);
		return cudaSuccess;
	}

	cudaError_t calculate_fractal_set(
		int* output,
		double x_min, double x_max,
		double y_min, double y_max,
		int width, int height,
		SET set_type,
		double julia_kr, double julia_ki,
		double mandelbox_fx, double mandelbox_sx, double mandelbox_rx)
	{
		int* dev_output = nullptr;
		cudaError_t cudaStatus;

		double dx = (x_max - x_min) / width;
		double dy = (y_max - y_min) / height;

		cudaStatus = cudaMalloc((void**)&dev_output, width * height * sizeof(int));
		if (cudaStatus != cudaSuccess)
			return cudaStatus;

		dim3 blockDim(16, 16);
		dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

		calculate_fractal << <gridDim, blockDim >> > (
			dev_output,
			x_min, y_min,
			dx, dy,
			width, height,
			set_type,
			julia_kr, julia_ki,
			mandelbox_fx, mandelbox_sx, mandelbox_rx
		);

		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess)
		{
			cudaFree(dev_output);
			return cudaStatus;
		}

		cudaStatus = cudaMemcpy(output, dev_output, width * height * sizeof(int), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess)
		{
			cudaFree(dev_output);
			return cudaStatus;
		}

		cudaFree(dev_output);
		return cudaSuccess;
	}
}
