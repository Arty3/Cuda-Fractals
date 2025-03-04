#pragma once

#include <math.h>

namespace sets
{
	constexpr const static inline int MAX_ITER = 80;
	static_assert(MAX_ITER > 0, "MAX_ITER must be greater than 0");

	enum class SET : __int8
	{
		MANDELBROT,
		MANDELBOX,
		TRICORN,
		JULIA,
	};

	int	mandelbrot(double cr, double ci)
	{
		int		n	= 0;
		double	x	= 0;
		double	y	= 0;
		double	x2	= 0;
		double	y2	= 0;

		while (x2 + y2 <= 4.0 && ++n < MAX_ITER + 1)
		{
			y = 2 * x * y + ci;
			x = x2 - y2 + cr;

			x2 = x * x;
			y2 = y * y;
		}
		return (n - 1);
	}

	static __forceinline double	box_fold(double v)
	{
		if (v > 1)
			v = 2 - v;
		else if (v < -1)
			v = -2 - v;

		return (v);
	}

	static __forceinline double	ball_fold(double r, double m)
	{
		if (m < r)
			m = m / (r * r);
		else if (m < 1)
			m = 1 / (m * m);

		return (m);
	}

	int	mandelbox(double fx, double sx, double rx, double cr, double ci)
	{
		int		n	= 0;
		double	vr	= cr;
		double	vi	= ci;
		double	mag	= 0;

		while (++n < MAX_ITER + 1)
		{
			vr = fx * box_fold(vr);
			vi = fx * box_fold(vi);

			mag = sqrt(vr * vr + vi * vi);

			vr = vr * sx * ball_fold(rx, mag) + cr;
			vi = vi * sx * ball_fold(rx, mag) + ci;

			if (sqrt(mag) > 2)
				break;
		}

		return (n - 1);
	}


	int	julia(double ki, double kr, double zr, double zi)
	{
		int		n = 0;
		double	tmp;

		while (++n < MAX_ITER + 1)
		{
			if ((zi * zi + zr * zr) > 4.0)
				break;

			tmp	= 2 * zr * zi + ki;
			zr	= zr * zr - zi * zi + kr;
			zi	= tmp;
		}

		return (n - 1);
	}

	int	tricorn(double cr, double ci)
	{
		int		n	= 0;
		double	zr	= cr;
		double	zi	= ci;

		double	tmp;

		while (++n < MAX_ITER + 1)
		{
			if ((zr * zr + zi * zi) > 4.0)
				break;

			tmp	= -2 * zr * zi + ci;
			zr	= zr * zr - zi * zi + cr;
			zi	= tmp;
		}

		return (n - 1);
	}
}
