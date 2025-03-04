#pragma once

#include <GLFW/glfw3.h>

class GLManager
{
private:
	bool internal_failure;

	GLFWwindow* win;
	GLFWcursor* cursor;

	int monitor_idx;
	int monitor_width;
	int monitor_height;

	void createWindow(void)
	{
		int count;

		GLFWmonitor** monitors = glfwGetMonitors(&count);

		if (monitor_idx > count || !monitors)
		{
			internal_failure = true;
			return;
		}

		win = glfwCreateWindow(
			monitor_width, monitor_height,
			"Fractal", monitors[monitor_idx], nullptr
		);
	}

	void closeWindow(void)
	{
		glfwDestroyWindow(win);
		glfwTerminate();
	}

	void createCursor(void)
	{
		cursor = glfwCreateStandardCursor(GLFW_ARROW_CURSOR);

		if (cursor)
		{
			glfwSetCursor(win, cursor);
			glfwSetInputMode(win, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
		}

		internal_failure = true;
	}

	void destroyCursor(void)
	{
		glfwSetInputMode(win, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
		glfwDestroyCursor(cursor);
	}

	void getMonitorSize(void)
	{
		monitor_width	= 0;
		monitor_height	= 0;

		int count;

		GLFWmonitor** monitors = glfwGetMonitors(&count);

		if (monitor_idx > count || !monitors)
		{
			internal_failure = true;
			return;
		}

		const GLFWvidmode* mode = glfwGetVideoMode(monitors[monitor_idx]);

		if (mode)
		{
			monitor_width	= mode->width;
			monitor_height	= mode->height;
		}

		internal_failure = true;
	}

	void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
	{
	}

	void cursorPosCallback(GLFWwindow* window, double xpos, double ypos)
	{
	}

	void scrollCallback(GLFWwindow* window, double xoffset, double yoffset)
	{
	}

	void resizeCallback(GLFWwindow* window, int width, int height)
	{
	}

public:
	GLManager(int monitor_idx = 0)
	{
		this->monitor_idx = monitor_idx;

		internal_failure = false;

		monitor_width	= 0;
		monitor_height	= 0;

		win		= nullptr;
		cursor	= nullptr;

		if (!glfwInit())
			goto fail;

		createWindow();
		createCursor();

		return;

	fail:
		internal_failure = true;
		return;
	}

	~GLManager(void)
	{
		closeWindow();
		destroyCursor();
	}

	__forceinline bool succeeded(void)
	{
		return !internal_failure;
	}

	void setMonitor(int idx)
	{
		monitor_idx = idx;
		getMonitorSize();
	}

	void getMonitorDimensions(int& width, int& height)
	{
		width	= monitor_width;
		height	= monitor_height;
	}

	__forceinline bool isKeyDown(int key)
	{
		return glfwGetKey(win, key) == GLFW_PRESS;
	}
};
