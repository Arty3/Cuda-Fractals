#pragma once

#include "Colors.hpp"
#include "Sets.hpp"

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
			_max_i = -1.5;
			_max_r = 1.0;

			_min_i = _max_i + (_max_r - _min_r) * h / w;
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

	__forceinline colors::STYLE getStyle(void) const
	{
		return _style;
	}

private:
	sets::SET		_set;
	colors::STYLE	_style;

	double			_min_r;
	double			_max_r;
	double			_min_i;
	double			_max_i;

	double			_kr;
	double			_ki;
	double			_sx;
	double			_rx;
	double			_fx;

	int*			_palette;
	int				_color;
};
