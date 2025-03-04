#pragma once

#include "Sets.cuh"

namespace view
{
	constexpr const static double MOVE_DISTANCE		= 0.05;
	constexpr const static double ZOOM_IN_FACTOR	= 0.8;
	constexpr const static double ZOOM_OUT_FACTOR	= 1.5;

	enum class DIRECTION : __int8
	{
		LEFT,
		RIGHT,
		UP,
		DOWN
	};

	struct ViewState
	{
		double min_r;
		double min_i;
		double max_r;
		double max_i;
	};

	cudaError_t zoom(
		int* output,
		ViewState& view,
		double zoom_factor,
		int width,
		int height,
		sets::SET fractal_type,
		double julia_kr = 0.0,
		double julia_ki = 0.0,
		double mandelbox_fx = 0.0,
		double mandelbox_sx = 0.0,
		double mandelbox_rx = 0.0);

	cudaError_t zoom_in(
		int* output,
		ViewState& view,
		int width,
		int height,
		sets::SET fractal_type,
		double julia_kr = 0.0,
		double julia_ki = 0.0,
		double mandelbox_fx = 0.0,
		double mandelbox_sx = 0.0,
		double mandelbox_rx = 0.0);

	cudaError_t zoom_out(
		int* output,
		ViewState& view,
		int width,
		int height,
		sets::SET fractal_type,
		double julia_kr = 0.0,
		double julia_ki = 0.0,
		double mandelbox_fx = 0.0,
		double mandelbox_sx = 0.0,
		double mandelbox_rx = 0.0);

	cudaError_t move(
		int* output,
		ViewState& view,
		DIRECTION direction,
		double distance_factor,
		int width,
		int height,
		sets::SET fractal_type,
		double julia_kr = 0.0,
		double julia_ki = 0.0,
		double mandelbox_fx = 0.0,
		double mandelbox_sx = 0.0,
		double mandelbox_rx = 0.0);

	cudaError_t move_standard(
		int* output,
		ViewState& view,
		DIRECTION direction,
		int width,
		int height,
		sets::SET fractal_type,
		double julia_kr = 0.0,
		double julia_ki = 0.0,
		double mandelbox_fx = 0.0,
		double mandelbox_sx = 0.0,
		double mandelbox_rx = 0.0);
}
