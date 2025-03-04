#pragma once

#include <cuda_runtime.h>

namespace sets
{
	constexpr const static int MAX_ITER = 80;
	static_assert(MAX_ITER > 0, "MAX_ITER must be greater than 0");

	enum class SET : __int8
	{
		MANDELBROT,
		MANDELBOX,
		TRICORN,
		JULIA,
	};

	__device__ int mandelbrot_kernel(double cr, double ci);
	__device__ double box_fold_kernel(double v);
	__device__ double ball_fold_kernel(double r, double m);
	__device__ int mandelbox_kernel(double fx, double sx, double rx, double cr, double ci);
	__device__ int julia_kernel(double ki, double kr, double zr, double zi);
	__device__ int tricorn_kernel(double cr, double ci);

	__global__ void calculate_fractal(
		int* output,
		double x_min, double y_min,
		double dx, double dy,
		int width, int height,
		SET set_type,
		double julia_kr, double julia_ki,
		double mandelbox_fx, double mandelbox_sx, double mandelbox_rx);

	cudaError_t calculate_mandelbrot(
		int* output,
		double x_min, double x_max,
		double y_min, double y_max,
		int width, int height);

	cudaError_t calculate_mandelbox(
		int* output,
		double x_min, double x_max,
		double y_min, double y_max,
		double fx, double sx, double rx,
		int width, int height);

	cudaError_t calculate_julia(
		int* output,
		double x_min, double x_max,
		double y_min, double y_max,
		double kr, double ki,
		int width, int height);

	cudaError_t calculate_tricorn(
		int* output,
		double x_min, double x_max,
		double y_min, double y_max,
		int width, int height);

	cudaError_t calculate_fractal_set(
		int* output,
		double x_min, double x_max,
		double y_min, double y_max,
		int width, int height,
		SET set_type,
		double julia_kr = 0.0, double julia_ki = 0.0,
		double mandelbox_fx = 0.0, double mandelbox_sx = 0.0, double mandelbox_rx = 0.0);
}
