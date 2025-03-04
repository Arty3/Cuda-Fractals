#pragma once

#include <GL/glew.h>
#include <iostream>
#include <fstream>
#include <string>

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include "Fractal.hpp"
#include "Colors.hpp"
#include "Sets.cuh"
#include "View.cuh"

constexpr const char* VERTEX_SHADER_PATH	= "colors_vertex.glsl";
constexpr const char* FRAGMENT_SHADER_PATH	= "colors_fragment.glsl";

class FractalRenderer
{
private:
	GLuint m_vao;
	GLuint m_vbo;
	GLuint m_ebo;
	GLuint m_texture;
	GLuint m_shader;

	cudaGraphicsResource* m_cudaTextureResource;

	int m_width;
	int m_height;

	int* m_iterations;

	view::ViewState m_viewState;

	Fractal& m_fractal;

	std::string loadShaderSource(const char* filePath)
	{
		std::string content;
		std::ifstream fileStream(filePath, std::ios::in);

		if (!fileStream.is_open())
		{
			std::cerr << "Could not read file " << filePath << ". File does not exist." << std::endl;
			return "";
		}

		std::string line = "";
		while (!fileStream.eof())
		{
			std::getline(fileStream, line);
			content.append(line + "\n");
		}

		fileStream.close();
		return content;
	}

	void compileShaders(void)
	{
		std::string vertexSource = loadShaderSource(VERTEX_SHADER_PATH);
		std::string fragmentSource = loadShaderSource(FRAGMENT_SHADER_PATH);

		const char* vertexSourcePtr = vertexSource.c_str();
		const char* fragmentSourcePtr = fragmentSource.c_str();

		GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
		glShaderSource(vertexShader, 1, &vertexSourcePtr, NULL);
		glCompileShader(vertexShader);

		int success;
		char infoLog[512];

		glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);

		if (!success)
		{
			glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
			std::cerr << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
		}

		GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
		glShaderSource(fragmentShader, 1, &fragmentSourcePtr, NULL);
		glCompileShader(fragmentShader);

		glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
		if (!success)
		{
			glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
			std::cerr << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
		}

		m_shader = glCreateProgram();
		glAttachShader(m_shader, vertexShader);
		glAttachShader(m_shader, fragmentShader);
		glLinkProgram(m_shader);

		glGetProgramiv(m_shader, GL_LINK_STATUS, &success);
		if (!success)
		{
			glGetProgramInfoLog(m_shader, 512, NULL, infoLog);
			std::cerr << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
		}

		glDeleteShader(vertexShader);
		glDeleteShader(fragmentShader);
	}

	void setupGL(void)
	{
		float vertices[] = {
			 1.0f,  1.0f, 0.0f, 1.0f, 1.0f,  // Top right
			 1.0f, -1.0f, 0.0f, 1.0f, 0.0f,  // Bottom right
			-1.0f, -1.0f, 0.0f, 0.0f, 0.0f,  // Bottom left
			-1.0f,  1.0f, 0.0f, 0.0f, 1.0f   // Top left
		};

		unsigned int indices[] = {
			0, 1, 3,  // First triangle
			1, 2, 3   // Second triangle
		};

		glGenVertexArrays(1, &m_vao);
		glBindVertexArray(m_vao);

		glGenBuffers(1, &m_vbo);
		glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

		glGenBuffers(1, &m_ebo);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_ebo);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
		glEnableVertexAttribArray(0);

		glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
		glEnableVertexAttribArray(1);

		glGenTextures(1, &m_texture);
		glBindTexture(GL_TEXTURE_2D, m_texture);

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

		glTexImage2D(GL_TEXTURE_2D, 0, GL_R32I, m_width, m_height, 0, GL_RED_INTEGER, GL_INT, NULL);

		cudaGraphicsGLRegisterImage(&m_cudaTextureResource, m_texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);

		glBindVertexArray(0);
	}

	void syncViewStateFromFractal()
	{
		m_viewState.min_r = m_fractal.getMinR();
		m_viewState.max_r = m_fractal.getMaxR();
		m_viewState.min_i = m_fractal.getMinI();
		m_viewState.max_i = m_fractal.getMaxI();
	}

	void syncFractalFromViewState()
	{
		m_fractal.setViewBounds(
			m_viewState.min_r,
			m_viewState.max_r,
			m_viewState.min_i,
			m_viewState.max_i
		);
	}

public:
	FractalRenderer(Fractal& fractal, int width, int height)
		: m_width(width), m_height(height), m_fractal(fractal)
	{
		m_iterations = new int[width * height];

		syncViewStateFromFractal();

		compileShaders();
		setupGL();

		calculateFractal();
	}

	~FractalRenderer()
	{
		cudaGraphicsUnregisterResource(m_cudaTextureResource);

		glDeleteVertexArrays(1, &m_vao);
		glDeleteBuffers(1, &m_vbo);
		glDeleteBuffers(1, &m_ebo);
		glDeleteTextures(1, &m_texture);
		glDeleteProgram(m_shader);

		delete[] m_iterations;
	}

	void calculateFractal()
	{
		cudaError_t status;

		syncViewStateFromFractal();

		switch (m_fractal.getSet())
		{
		case sets::SET::MANDELBROT:
			status = sets::calculate_mandelbrot(
				m_iterations,
				m_viewState.min_r, m_viewState.max_r,
				m_viewState.min_i, m_viewState.max_i,
				m_width, m_height);
			break;

		case sets::SET::MANDELBOX:
			status = sets::calculate_mandelbox(
				m_iterations,
				m_viewState.min_r, m_viewState.max_r,
				m_viewState.min_i, m_viewState.max_i,
				m_fractal.getMandelboxFx(), m_fractal.getMandelboxSx(), m_fractal.getMandelboxRx(),
				m_width, m_height);
			break;

		case sets::SET::JULIA:
			status = sets::calculate_julia(
				m_iterations,
				m_viewState.min_r, m_viewState.max_r,
				m_viewState.min_i, m_viewState.max_i,
				m_fractal.getJuliaKr(), m_fractal.getJuliaKi(),
				m_width, m_height);
			break;

		case sets::SET::TRICORN:
			status = sets::calculate_tricorn(
				m_iterations,
				m_viewState.min_r, m_viewState.max_r,
				m_viewState.min_i, m_viewState.max_i,
				m_width, m_height);
			break;
		default:
			status = cudaErrorUnknown;
			break;
		}

		if (status != cudaSuccess)
			std::cerr << "CUDA error: " << cudaGetErrorString(status) << std::endl;

		glBindTexture(GL_TEXTURE_2D, m_texture);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_width, m_height, GL_RED_INTEGER, GL_INT, m_iterations);
	}

	void zoom(double factor)
	{
		cudaError_t status = view::zoom(
			m_iterations,
			m_viewState,
			factor,
			m_width, m_height,
			m_fractal.getSet(),
			m_fractal.getJuliaKr(), m_fractal.getJuliaKi(),
			m_fractal.getMandelboxFx(), m_fractal.getMandelboxSx(), m_fractal.getMandelboxRx()
		);

		if (status != cudaSuccess)
			std::cerr << "CUDA error: " << cudaGetErrorString(status) << std::endl;

		syncFractalFromViewState();

		glBindTexture(GL_TEXTURE_2D, m_texture);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_width, m_height, GL_RED_INTEGER, GL_INT, m_iterations);
	}

	void move(view::DIRECTION direction)
	{
		cudaError_t status = view::move_standard(
			m_iterations,
			m_viewState,
			direction,
			m_width, m_height,
			m_fractal.getSet(),
			m_fractal.getJuliaKr(), m_fractal.getJuliaKi(),
			m_fractal.getMandelboxFx(), m_fractal.getMandelboxSx(), m_fractal.getMandelboxRx()
		);

		if (status != cudaSuccess)
			std::cerr << "CUDA error: " << cudaGetErrorString(status) << std::endl;

		syncFractalFromViewState();

		glBindTexture(GL_TEXTURE_2D, m_texture);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_width, m_height, GL_RED_INTEGER, GL_INT, m_iterations);
	}

	void resize(int width, int height)
	{
		if (m_width == width && m_height == height)
			return;

		m_width = width;
		m_height = height;

		delete[] m_iterations;
		m_iterations = new int[width * height];

		glBindTexture(GL_TEXTURE_2D, m_texture);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_R32I, m_width, m_height, 0, GL_RED_INTEGER, GL_INT, NULL);

		double current_width = m_viewState.max_r - m_viewState.min_r;
		double current_height = m_viewState.max_i - m_viewState.min_i;
		double center_r = (m_viewState.min_r + m_viewState.max_r) / 2.0;
		double center_i = (m_viewState.min_i + m_viewState.max_i) / 2.0;

		double new_height = current_width * height / width;

		m_viewState.min_i = center_i - new_height / 2.0;
		m_viewState.max_i = center_i + new_height / 2.0;

		syncFractalFromViewState();
		calculateFractal();
	}

	void render()
	{
		glUseProgram(m_shader);

		glUniform1i(glGetUniformLocation(m_shader, "u_maxIterations"), sets::MAX_ITER);
		glUniform1i(glGetUniformLocation(m_shader, "u_colorStyle"), static_cast<int>(m_fractal.getStyle()));
		glUniform1i(glGetUniformLocation(m_shader, "u_baseColor"), m_fractal.getColor());

		int* palette = m_fractal.getPalette();
		int colorArray[4];

		for (int i = 0; i < 4; ++i)
			colorArray[i] = palette[i * sets::MAX_ITER / 4];

		glUniform1iv(glGetUniformLocation(m_shader, "u_colorArray"), 4, colorArray);
		glUniform1i(glGetUniformLocation(m_shader, "u_colorCount"), 4);

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, m_texture);
		glUniform1i(glGetUniformLocation(m_shader, "u_iterationTexture"), 0);

		glBindVertexArray(m_vao);
		glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
		glBindVertexArray(0);
	}
};
