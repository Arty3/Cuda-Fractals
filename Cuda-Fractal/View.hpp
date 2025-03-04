#pragma once

namespace view
{
	constexpr const static inline double	MOVE_DISTANCE	= 0.05;
	constexpr const static inline double	ZOOM_IN_FACTOR	= 0.8;
	constexpr const static inline double	ZOOM_OUT_FACTOR	= 1.5;

	enum class DIRECTION : __int8
	{
		LEFT,
		RIGHT,
		UP,
		DOWN
	};

	void zoom(double min_r, double min_i, double max_r, double max_i, double zoom)
	{
		const double center_r = min_r - max_r;
		const double center_i = max_i - min_i;

		max_r = max_r + (center_r - zoom * center_r) / 2;
		min_r = max_r + zoom * center_r;
		min_i = min_i + (center_i - zoom * center_i) / 2;
		max_i = min_i + zoom * center_i;

		// Render call here
	}

	void move(double min_r, double min_i, double max_r, double max_i, double distance, DIRECTION direction)
	{
		const double center_r = max_r - min_r;
		const double center_i = max_i - min_i;

		if (direction == DIRECTION::LEFT)
		{
			min_r -= center_r * distance;
			max_r -= center_r * distance;
		}
		else if (direction == DIRECTION::RIGHT)
		{
			min_r += center_r * distance;
			max_r += center_r * distance;
		}
		else if (direction == DIRECTION::UP)
		{
			min_i += center_i * distance;
			max_i += center_i * distance;
		}
		else if (direction == DIRECTION::DOWN)
		{
			min_i -= center_i * distance;
			max_i -= center_i * distance;
		}
		
		// Render call here
	}

}
