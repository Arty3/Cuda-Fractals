#include "View.cuh"

namespace view
{
	cudaError_t zoom(
		int* output,
		ViewState& view,
		double zoom_factor,
		int width,
		int height,
		sets::SET fractal_type,
		double julia_kr,
		double julia_ki,
		double mandelbox_fx,
		double mandelbox_sx,
		double mandelbox_rx)
	{
		double center_r = (view.max_r + view.min_r) / 2.0;
		double center_i = (view.max_i + view.min_i) / 2.0;

		double width_r = (view.max_r - view.min_r) * zoom_factor;
		double height_i = (view.max_i - view.min_i) * zoom_factor;

		view.min_r = center_r - width_r / 2.0;
		view.max_r = center_r + width_r / 2.0;
		view.min_i = center_i - height_i / 2.0;
		view.max_i = center_i + height_i / 2.0;

		return sets::calculate_fractal_set(
			output,
			view.min_r, view.max_r,
			view.min_i, view.max_i,
			width, height,
			fractal_type,
			julia_kr, julia_ki,
			mandelbox_fx, mandelbox_sx, mandelbox_rx
		);
	}

	cudaError_t zoom_in(
		int* output,
		ViewState& view,
		int width,
		int height,
		sets::SET fractal_type,
		double julia_kr,
		double julia_ki,
		double mandelbox_fx,
		double mandelbox_sx,
		double mandelbox_rx)
	{
		return zoom(
			output,
			view,
			ZOOM_IN_FACTOR,
			width,
			height,
			fractal_type,
			julia_kr,
			julia_ki,
			mandelbox_fx,
			mandelbox_sx,
			mandelbox_rx
		);
	}

	cudaError_t zoom_out(
		int* output,
		ViewState& view,
		int width,
		int height,
		sets::SET fractal_type,
		double julia_kr,
		double julia_ki,
		double mandelbox_fx,
		double mandelbox_sx,
		double mandelbox_rx)
	{
		return zoom(
			output,
			view,
			ZOOM_OUT_FACTOR,
			width,
			height,
			fractal_type,
			julia_kr,
			julia_ki,
			mandelbox_fx,
			mandelbox_sx,
			mandelbox_rx
		);
	}

	cudaError_t move(
		int* output,
		ViewState& view,
		DIRECTION direction,
		double distance_factor,
		int width,
		int height,
		sets::SET fractal_type,
		double julia_kr,
		double julia_ki,
		double mandelbox_fx,
		double mandelbox_sx,
		double mandelbox_rx)
	{
		double width_r = view.max_r - view.min_r;
		double height_i = view.max_i - view.min_i;
		double distance_r = width_r * distance_factor;
		double distance_i = height_i * distance_factor;

		switch (direction)
		{
		case DIRECTION::LEFT:
			view.min_r -= distance_r;
			view.max_r -= distance_r;
			break;
		case DIRECTION::RIGHT:
			view.min_r += distance_r;
			view.max_r += distance_r;
			break;
		case DIRECTION::UP:
			view.min_i += distance_i;
			view.max_i += distance_i;
			break;
		case DIRECTION::DOWN:
			view.min_i -= distance_i;
			view.max_i -= distance_i;
			break;
		}

		return sets::calculate_fractal_set(
			output,
			view.min_r, view.max_r,
			view.min_i, view.max_i,
			width, height,
			fractal_type,
			julia_kr, julia_ki,
			mandelbox_fx, mandelbox_sx, mandelbox_rx
		);
	}

	cudaError_t move_standard(
		int* output,
		ViewState& view,
		DIRECTION direction,
		int width,
		int height,
		sets::SET fractal_type,
		double julia_kr,
		double julia_ki,
		double mandelbox_fx,
		double mandelbox_sx,
		double mandelbox_rx)
	{
		return move(
			output,
			view,
			direction,
			MOVE_DISTANCE,
			width,
			height,
			fractal_type,
			julia_kr,
			julia_ki,
			mandelbox_fx,
			mandelbox_sx,
			mandelbox_rx
		);
	}
}
