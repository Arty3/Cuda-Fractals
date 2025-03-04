#include "GLManager.hpp"
#include "Fractal.hpp"
#include "Renderer.hpp"
#include "View.cuh"

#include <iostream>
#include <string>
#include <limits>

struct ApplicationState
{
	GLManager*			glManager;
	Fractal*			fractal;
	FractalRenderer*	renderer;
	double				cursor_x;
	double				cursor_y;
	int					width;
	int					height;
};

void cleanup(ApplicationState* state);

sets::SET select_fractal_set()
{
	int selection = 0;
	bool valid_input = false;

	while (!valid_input)
	{
		std::cout << "\n===== FRACTAL SET SELECTION =====" << std::endl;
		std::cout << "1. Mandelbrot Set" << std::endl;
		std::cout << "2. Julia Set" << std::endl;
		std::cout << "3. Tricorn Set" << std::endl;
		std::cout << "4. Mandelbox Set" << std::endl;
		std::cout << "Enter your selection (1-4): ";

		if (std::cin >> selection)
		{
			if (selection >= 1 && selection <= 4)
			{
				valid_input = true;
			}
			else
			{
				std::cout << "Error: Invalid selection. Please enter a number between 1 and 4." << std::endl;
			}
		}
		else
		{
			std::cin.clear();
			std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
			std::cout << "Error: Invalid input. Please enter a number." << std::endl;
		}
	}

	switch (selection)
	{
	case 1:
		std::cout << "Selected: Mandelbrot Set" << std::endl;
		return sets::SET::MANDELBROT;
	case 2:
		std::cout << "Selected: Julia Set" << std::endl;
		return sets::SET::JULIA;
	case 3:
		std::cout << "Selected: Tricorn Set" << std::endl;
		return sets::SET::TRICORN;
	case 4:
		std::cout << "Selected: Mandelbox Set" << std::endl;
		return sets::SET::MANDELBOX;
	default:
		// Fallback (shouldn't happen due to validation)
		std::cout << "Default: Mandelbrot Set" << std::endl;
		return sets::SET::MANDELBROT;
	}
}

colors::STYLE select_color_style()
{
	int selection = 0;
	bool valid_input = false;

	while (!valid_input)
	{
		std::cout << "\n===== COLOR STYLE SELECTION =====" << std::endl;
		std::cout << "1. Mono (Gradient)" << std::endl;
		std::cout << "2. Multiple (Multi-colored gradients)" << std::endl;
		std::cout << "3. Opposite (Contrasting colors)" << std::endl;
		std::cout << "4. Contrasted (High contrast colors)" << std::endl;
		std::cout << "5. Graphic (Design style)" << std::endl;
		std::cout << "6. Zebra (Alternating stripes)" << std::endl;
		std::cout << "7. Triad (Three-color stripes)" << std::endl;
		std::cout << "8. Tetra (Four-color stripes)" << std::endl;
		std::cout << "Enter your selection (1-8): ";

		if (std::cin >> selection)
		{
			if (selection >= 1 && selection <= 8)
			{
				valid_input = true;
			}
			else
			{
				std::cout << "Error: Invalid selection. Please enter a number between 1 and 8." << std::endl;
			}
		}
		else
		{
			std::cin.clear();
			std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
			std::cout << "Error: Invalid input. Please enter a number." << std::endl;
		}
	}

	colors::STYLE style;
	switch (selection)
	{
	case 1:
		std::cout << "Selected: Mono style" << std::endl;
		return colors::STYLE::MONO;
	case 2:
		std::cout << "Selected: Multiple style" << std::endl;
		return colors::STYLE::MULTIPLE;
	case 3:
		std::cout << "Selected: Opposite style" << std::endl;
		return colors::STYLE::OPPOSITE;
	case 4:
		std::cout << "Selected: Contrasted style" << std::endl;
		return colors::STYLE::CONTRASTED;
	case 5:
		std::cout << "Selected: Graphic style" << std::endl;
		return colors::STYLE::GRAPHIC;
	case 6:
		std::cout << "Selected: Zebra style" << std::endl;
		return colors::STYLE::ZEBRA;
	case 7:
		std::cout << "Selected: Triad style" << std::endl;
		return colors::STYLE::TRIAD;
	case 8:
		std::cout << "Selected: Tetra style" << std::endl;
		return colors::STYLE::TETRA;
	default:
		std::cout << "Default: Mono style" << std::endl;
		return colors::STYLE::MONO;
	}
}

int select_base_color()
{
	int selection = 0;
	bool valid_input = false;

	while (!valid_input)
	{
		std::cout << "\n===== BASE COLOR SELECTION =====" << std::endl;
		std::cout << "1. Blue (0x0000FF)" << std::endl;
		std::cout << "2. Red (0xFF0000)" << std::endl;
		std::cout << "3. Green (0x00FF00)" << std::endl;
		std::cout << "4. Yellow (0xFFFF00)" << std::endl;
		std::cout << "5. Cyan (0x00FFFF)" << std::endl;
		std::cout << "6. Magenta (0xFF00FF)" << std::endl;
		std::cout << "7. Orange (0xFF8800)" << std::endl;
		std::cout << "8. Purple (0x8800FF)" << std::endl;
		std::cout << "9. White (0xFFFFFF)" << std::endl;
		std::cout << "10. Black (0x000000)" << std::endl;
		std::cout << "Enter your selection (1-10): ";

		if (std::cin >> selection)
		{
			if (selection >= 1 && selection <= 10)
			{
				valid_input = true;
			}
			else
			{
				std::cout << "Error: Invalid selection. Please enter a number between 1 and 10." << std::endl;
			}
		}
		else
		{
			std::cin.clear();
			std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
			std::cout << "Error: Invalid input. Please enter a number." << std::endl;
		}
	}

	switch (selection)
	{
	case 1:
		std::cout << "Selected: Blue" << std::endl;
		return 0x0000FF;
	case 2:
		std::cout << "Selected: Red" << std::endl;
		return 0xFF0000;
	case 3:
		std::cout << "Selected: Green" << std::endl;
		return 0x00FF00;
	case 4:
		std::cout << "Selected: Yellow" << std::endl;
		return 0xFFFF00;
	case 5:
		std::cout << "Selected: Cyan" << std::endl;
		return 0x00FFFF;
	case 6:
		std::cout << "Selected: Magenta" << std::endl;
		return 0xFF00FF;
	case 7:
		std::cout << "Selected: Orange" << std::endl;
		return 0xFF8800;
	case 8:
		std::cout << "Selected: Purple" << std::endl;
		return 0x8800FF;
	case 9:
		std::cout << "Selected: White" << std::endl;
		return 0xFFFFFF;
	case 10:
		std::cout << "Selected: Black" << std::endl;
		return 0x000000;
	default:
		std::cout << "Default: Blue" << std::endl;
		return 0x0000FF;
	}
}

void key_callback(int key, int scancode, int action, int mods)
{
	ApplicationState* state = reinterpret_cast<ApplicationState*>(glfwGetWindowUserPointer(glfwGetCurrentContext()));

	if (action != GLFW_PRESS && action != GLFW_REPEAT)
		return;

	switch (key)
	{
	case GLFW_KEY_UP:
		state->renderer->move(view::DIRECTION::UP);
		break;

	case GLFW_KEY_DOWN:
		state->renderer->move(view::DIRECTION::DOWN);
		break;

	case GLFW_KEY_LEFT:
		state->renderer->move(view::DIRECTION::LEFT);
		break;

	case GLFW_KEY_RIGHT:
		state->renderer->move(view::DIRECTION::RIGHT);
		break;

	case GLFW_KEY_SPACE:
		state->fractal->setStyle(static_cast<colors::STYLE>(
			(static_cast<int>(state->fractal->getStyle()) + 1) % 8));
		state->renderer->calculateFractal();
		break;

	case GLFW_KEY_KP_ADD:
	case GLFW_KEY_EQUAL:
		state->renderer->zoom(view::ZOOM_IN_FACTOR);
		break;

	case GLFW_KEY_KP_SUBTRACT:
	case GLFW_KEY_MINUS:
		state->renderer->zoom(view::ZOOM_OUT_FACTOR);
		break;

	case GLFW_KEY_ESCAPE:
		glfwSetWindowShouldClose(glfwGetCurrentContext(), GLFW_TRUE);
		break;
	}
}

void scroll_callback(double xoffset, double yoffset)
{
	ApplicationState* state = reinterpret_cast<ApplicationState*>(
		glfwGetWindowUserPointer(glfwGetCurrentContext()));

	if (yoffset > 0)
	{
		state->renderer->zoom(view::ZOOM_IN_FACTOR);

		double offset_x = state->cursor_x - state->width / 2;
		double offset_y = state->cursor_y - state->height / 2;

		if (offset_x < 0)
			state->renderer->move(view::DIRECTION::LEFT);
		else if (offset_x > 0)
			state->renderer->move(view::DIRECTION::RIGHT);

		if (offset_y < 0)
			state->renderer->move(view::DIRECTION::UP);
		else if (offset_y > 0)
			state->renderer->move(view::DIRECTION::DOWN);
	}
	else if (yoffset < 0)
	{
		state->renderer->zoom(view::ZOOM_OUT_FACTOR);
	}
}

void cursor_pos_callback(double xpos, double ypos)
{
	ApplicationState* state = reinterpret_cast<ApplicationState*>(
		glfwGetWindowUserPointer(glfwGetCurrentContext()));

	state->cursor_x = xpos;
	state->cursor_y = ypos;
}

void resize_callback(int width, int height)
{
	ApplicationState* state = reinterpret_cast<ApplicationState*>(
		glfwGetWindowUserPointer(glfwGetCurrentContext()));

	state->width = width;
	state->height = height;

	glViewport(0, 0, width, height);

	state->renderer->resize(width, height);
}

void cleanup(ApplicationState* state)
{
	if (state->renderer)
		delete state->renderer;
	if (state->fractal)
		delete state->fractal;
	if (state->glManager)
		delete state->glManager;
}

int main(int argc, char* argv[])
{
	ApplicationState state;

	sets::SET selectedSet		= select_fractal_set();
	colors::STYLE selectedStyle	= select_color_style();
	int selectedColor			= select_base_color();

	std::cout << "\nInitializing fractal explorer with your selections..." << std::endl;

	state.glManager = new GLManager(0);

	int w, h;

	state.glManager->getMonitorDimensions(&w, &h);

	state.width		= w;
	state.height	= h;
	state.cursor_x	= w / 2;
	state.cursor_y	= h / 2;

	if (!state.glManager->succeeded())
	{
		std::cerr << "Failed to initialize GLFW" << std::endl;
		cleanup(&state);
		return EXIT_FAILURE;
	}

	glfwSetWindowUserPointer(state.glManager->getWindow(), &state);

	state.fractal = new Fractal(
		selectedSet, selectedStyle, w, h
	);

	state.fractal->setColor(selectedColor);

	if (selectedSet == sets::SET::JULIA)
		state.fractal->setJuliaParams(-0.8, 0.156);

	else if (selectedSet == sets::SET::MANDELBOX)
		state.fractal->setMandelboxParams(2.0, 2.0, 0.5);

	if (glewInit() != GLEW_OK)
	{
		std::cerr << "Failed to initialize GLEW" << std::endl;
		cleanup(&state);
		return EXIT_FAILURE;
	}

	state.renderer = new FractalRenderer(*state.fractal, w, h);

	state.glManager->setKeyCallback(key_callback);
	state.glManager->setScrollCallback(scroll_callback);
	state.glManager->setCursorPosCallback(cursor_pos_callback);
	state.glManager->setResizeCallback(resize_callback);

	while (!state.glManager->shouldClose())
	{
		glClear(GL_COLOR_BUFFER_BIT);

		state.renderer->render();
		state.glManager->swapBuffers();
		state.glManager->pollEvents();
	}

	cleanup(&state);

	return EXIT_SUCCESS;
}
