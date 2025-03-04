#pragma once

#include "Sets.hpp"

/* Send to cuda later */

namespace colors
{
	enum class STYLE : __int8
	{
		MONO,
		MULTIPLE,
		OPPOSITE,
		CONTRASTED,
		GRAPHIC,
		ZEBRA,
		TRIAD,
		TETRA
	};

	namespace interpolated
	{
		static __forceinline int interpolate(int startcolor, int endcolor, double f)
		{
			int	start_rgb[3];
			int	end_rgb[3];

			start_rgb[0]	= ((startcolor >> 16)	& 0xFF);
			start_rgb[1]	= ((startcolor >> 8)	& 0xFF);
			start_rgb[2]	= ((startcolor >> 0)	& 0xFF);

			end_rgb[0]		= ((endcolor >> 16)		& 0xFF);
			end_rgb[1]		= ((endcolor >> 8)		& 0xFF);
			end_rgb[2]		= ((endcolor >> 0)		& 0xFF);

			start_rgb[0]	= (end_rgb[0] - start_rgb[0]) * f + start_rgb[0];
			start_rgb[1]	= (end_rgb[1] - start_rgb[1]) * f + start_rgb[1];
			start_rgb[2]	= (end_rgb[2] - start_rgb[2]) * f + start_rgb[2];

			return (0xFF << 24 | start_rgb[0] << 16 | start_rgb[1] << 8 | start_rgb[2]);
		}

		void setColorMono(int*& palette, int color)
		{
			int col1 = 0x000000;
			int col2 = color;

			for (int i = 0; i < sets::MAX_ITER; i += sets::MAX_ITER / 2)
			{
				for (int j = 0; j < sets::MAX_ITER / 2; ++j)
				{
					double f = static_cast<double>(j) / (sets::MAX_ITER / 2);
					palette[i + j] = interpolate(col1, col2, f);
				}
				col1 = col2;
				col2 = 0xFFFFFF;
			}
			palette[sets::MAX_ITER - 1] = 0;
		}

		void setColorMultiple(int*& palette, int colors[4], int n)
		{
			for (int i = 0, x = 0; i < sets::MAX_ITER; i += sets::MAX_ITER / (n - 1), ++x)
			{
				for (int j = 0; (i + j) < sets::MAX_ITER && j < (sets::MAX_ITER / (n - 1)); ++j)
				{
					double f = static_cast<double>(j) / (sets::MAX_ITER / (n - 1));
					palette[i + j] = interpolate(colors[x], colors[x + 1], f);
				}
			}

			if (sets::MAX_ITER)
				palette[sets::MAX_ITER - 1] = 0;
		}
	}

	namespace special
	{
		void setColorOpposites(int*& palette, int color)
		{
			int	r = (color >> 16)	& 0xFF;
			int	g = (color >> 8)	& 0xFF;
			int	b = (color >> 0)	& 0xFF;

			for (int i = 0; i < sets::MAX_ITER; ++i)
			{
				palette[i] = 0;

				r += i % 0xFF;
				g += i % 0xFF;
				b += i % 0xFF;

				palette[i] = 0xFF << 24 | r << 16 | g << 8 | b;
			}

			palette[sets::MAX_ITER - 1] = 0;
		}

		void setColorContrasted(int*& palette, int color)
		{
			int r = (color >> 16)	& 0xFF;
			int g = (color >> 8)	& 0xFF;
			int b = (color >> 0)	& 0xFF;

			for (int i = 0; i < sets::MAX_ITER; ++i)
			{
				palette[i] = 0;

				if (r != 0xFF)
					r += i % 0xFF;
				if (g != 0xFF)
					g += i % 0xFF;
				if (b != 0xFF)
					b += i % 0xFF;

				palette[i] = 0xFF << 24 | r << 16 | g << 8 | b;
			}

			palette[sets::MAX_ITER - 1] = 0;
		}

		void setColorGraphic(int*& palette, int color)
		{
			int r = (color >> 16) & 0xFF;
			int g = (color >> 8) & 0xFF;
			int b = (color >> 0) & 0xFF;

			while (r < 0x33 || g < 0x33 || b < 0x33)
			{
				if (r != 0xFF)
					++r;
				if (g != 0xFF)
					++g;
				if (b != 0xFF)
					++b;
			}

			for (int i = 0; i < sets::MAX_ITER; ++i)
			{
				palette[i] = 0;

				r -= i % 0xFF;
				g -= i % 0xFF;
				b -= i % 0xFF;

				palette[i] = 0xFF << 24 | r << 16 | g << 8 | b;
			}

			palette[sets::MAX_ITER - 1] = 0;
		}
	}

	namespace striped
	{
		static inline void	fillColorStripe(int*& palette, int color, int stripe)
		{
			for (int i = 0; i < sets::MAX_ITER; i += stripe)
				palette[i] = color;
		}

		__forceinline int getPercentColor(int color, double percent)
		{
			const double percentage = (percent / 100) * 256;

			int	rgb[3];
			int	trgb[3];

			rgb[0] = (color >> 16)	& 0xFF;
			rgb[1] = (color >> 8)	& 0xFF;
			rgb[2] = (color >> 0)	& 0xFF;

			trgb[0] = (rgb[0] + percentage) - 256;
			trgb[1] = (rgb[1] + percentage) - 256;
			trgb[2] = (rgb[2] + percentage) - 256;

			return (0xFF << 24 | trgb[0] << 16 | trgb[1] << 8 | trgb[2]);
		}

		void setColorZebra(int*& palette, int color)
		{
			const int c = getPercentColor(color, 50);

			fillColorStripe(palette, color, 1);
			fillColorStripe(palette, c, 2);

			palette[sets::MAX_ITER - 1] = 0;
		}

		void setColorTriad(int*& palette, int color)
		{
			int	triad[2];

			triad[0] = getPercentColor(color, 33);
			triad[1] = getPercentColor(color, 66);

			fillColorStripe(palette, color, 1);
			fillColorStripe(palette, triad[0], 2);
			fillColorStripe(palette, triad[1], 3);

			palette[sets::MAX_ITER - 1] = 0;
		}

		void setColorTetra(int*& palette, int color)
		{
			int	tetra[3];

			tetra[0] = getPercentColor(color, 25);
			tetra[1] = getPercentColor(color, 50);
			tetra[2] = getPercentColor(color, 75);

			fillColorStripe(palette, color, 1);
			fillColorStripe(palette, tetra[0], 2);
			fillColorStripe(palette, tetra[1], 3);
			fillColorStripe(palette, tetra[2], 4);

			palette[sets::MAX_ITER - 1] = 0;
		}
	}
}
