#pragma once

#include "Colors.hpp"
#include "Sets.cuh"

#include <cstring>

class Fractal
{
public:
	Fractal(sets::SET set, colors::STYLE style, int w, int h)
	{
		// Ugly but saves me sanity
		std::memset(this, 0, sizeof(*this));

		this->_set		= set;
		this->_style	= style;

		_palette = new int[sets::MAX_ITER + 1];

		if (set == sets::SET::MANDELBOX)
		{
			_min_r = -4.0;
			_min_i = -4.0;
			_max_r = 4.0;
			_max_i = _min_i + (_max_r - _min_r) * h / w;
		}
		else if (set == sets::SET::JULIA)
		{
			_min_r = -2.0;
			_min_i = -2.0;
			_max_r = 2.0;
			_max_i = _min_i + (_max_r - _min_r) * h / w;
		}
		else
		{
			_min_r = -2.0;
			_max_i = 1.5;
			_max_r = 1.0;
			_min_i = _max_i - (_max_r - _min_r) * h / w;
		}
	}

	~Fractal(void)
	{
		delete[] _palette;
	}

	__forceinline int*& getPalette(void)
	{
		return _palette;
	}

	__forceinline sets::SET getSet(void) const
	{
		return _set;
	}

	__forceinline double getMinR(void) const
	{
		return _min_r;
	}

	__forceinline double getMaxR(void) const
	{
		return _max_r;
	}

	__forceinline double getMinI(void) const
	{
		return _min_i;
	}

	__forceinline double getMaxI(void) const
	{
		return _max_i;
	}

	__forceinline colors::STYLE getStyle(void) const
	{
		return _style;
	}

	__forceinline void setStyle(colors::STYLE style)
	{
		_style = style;
	}

	__forceinline void setViewBounds(double min_r, double max_r, double min_i, double max_i)
	{
		_min_r = min_r;
		_max_r = max_r;
		_min_i = min_i;
		_max_i = max_i;
	}

	__forceinline void setJuliaParams(double kr, double ki)
	{
		_kr = kr;
		_ki = ki;
	}

	__forceinline void setMandelboxParams(double fx, double sx, double rx)
	{
		_fx = fx;
		_sx = sx;
		_rx = rx;
	}

	__forceinline double getJuliaKr(void) const
	{
		return _kr;
	}

	__forceinline double getJuliaKi(void) const
	{
		return _ki;
	}

	__forceinline double getMandelboxFx(void) const
	{
		return _fx;
	}

	__forceinline double getMandelboxSx(void) const
	{
		return _sx;
	}

	__forceinline double getMandelboxRx(void) const
	{
		return _rx;
	}

	__forceinline void setColor(int color)
	{
		_color = color;
	}

	__forceinline int getColor(void) const
	{
		return _color;
	}

private:
	sets::SET		_set;
	colors::STYLE	_style;

	double	_min_r;
	double	_max_r;
	double	_min_i;
	double	_max_i;

	double	_kr;
	double	_ki;
	double	_sx;
	double	_rx;
	double	_fx;

	int*	_palette;
	int		_color;
};
