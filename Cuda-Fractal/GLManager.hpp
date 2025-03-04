#pragma once

#include <GLFW/glfw3.h>
#include <functional>

class GLManager
{
private:
	bool internal_failure;

	GLFWwindow* win;
	GLFWcursor* cursor;

	int monitor_idx;
	int monitor_width;
	int monitor_height;

	std::function<void(int, int, int, int)>	key_callback_fn;
	std::function<void(double, double)>		cursor_pos_callback_fn;
	std::function<void(double, double)>		scroll_callback_fn;
	std::function<void(int, int)>			resize_callback_fn;

	static void key_callback_static(GLFWwindow* window, int key, int scancode, int action, int mods)
	{
		GLManager* manager = static_cast<GLManager*>(glfwGetWindowUserPointer(window));

		if (manager && manager->key_callback_fn)
			manager->key_callback_fn(key, scancode, action, mods);
	}

	static void cursor_pos_callback_static(GLFWwindow* window, double xpos, double ypos)
	{
		GLManager* manager = static_cast<GLManager*>(glfwGetWindowUserPointer(window));

		if (manager && manager->cursor_pos_callback_fn)
			manager->cursor_pos_callback_fn(xpos, ypos);
	}

	static void scroll_callback_static(GLFWwindow* window, double xoffset, double yoffset)
	{
		GLManager* manager = static_cast<GLManager*>(glfwGetWindowUserPointer(window));

		if (manager && manager->scroll_callback_fn)
			manager->scroll_callback_fn(xoffset, yoffset);
	}

	static void resize_callback_static(GLFWwindow* window, int width, int height)
	{
		GLManager* manager = static_cast<GLManager*>(glfwGetWindowUserPointer(window));

		if (manager && manager->resize_callback_fn)
			manager->resize_callback_fn(width, height);
	}

	void createWindow(void)
	{
		int count;
		GLFWmonitor** monitors = glfwGetMonitors(&count);

		if (monitor_idx >= count || !monitors)
		{
			internal_failure = true;
			return;
		}

		if (!monitor_width || !monitor_height)
			getMonitorSize();

		win = glfwCreateWindow(
			monitor_width, monitor_height, "Fractal Explorer",
			monitors[monitor_idx], nullptr
		);

		if (!win)
		{
			internal_failure = true;
			return;
		}

		glfwSetWindowUserPointer(win, this);

		glfwSetKeyCallback(win, key_callback_static);
		glfwSetCursorPosCallback(win, cursor_pos_callback_static);
		glfwSetScrollCallback(win, scroll_callback_static);
		glfwSetFramebufferSizeCallback(win, resize_callback_static);

		glfwMakeContextCurrent(win);
	}

	void closeWindow(void)
	{
		if (win)
		{
			glfwDestroyWindow(win);
			win = nullptr;
		}
		glfwTerminate();
	}

	void createCursor(void)
	{
		cursor = glfwCreateStandardCursor(GLFW_ARROW_CURSOR);
		if (cursor && win)
		{
			glfwSetCursor(win, cursor);
			glfwSetInputMode(win, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
		}
		else
		{
			internal_failure = true;
		}
	}

	void destroyCursor(void)
	{
		if (win && cursor)
		{
			glfwSetInputMode(win, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
			glfwDestroyCursor(cursor);
			cursor = nullptr;
		}
	}

	void getMonitorSize(void)
	{
		monitor_width = 0;
		monitor_height = 0;

		int count;
		GLFWmonitor** monitors = glfwGetMonitors(&count);

		if (monitor_idx >= count || !monitors)
		{
			internal_failure = true;
			return;
		}

		const GLFWvidmode* mode = glfwGetVideoMode(monitors[monitor_idx]);

		if (mode)
		{
			monitor_width = mode->width;
			monitor_height = mode->height;
		}
		else
		{
			internal_failure = true;
		}
	}

public:
	GLManager(int monitor_idx = 0)
	{
		this->monitor_idx	= monitor_idx;
		internal_failure	= false;
		monitor_width		= 0;
		monitor_height		= 0;
		win					= nullptr;
		cursor				= nullptr;

		if (!glfwInit())
		{
			internal_failure = true;
			return;
		}

		glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
		glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
		glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

		getMonitorSize();
		createWindow();
		createCursor();
	}

	~GLManager(void)
	{
		destroyCursor();
		closeWindow();
	}

	__forceinline bool succeeded(void)
	{
		return !internal_failure && win != nullptr;
	}

	__forceinline GLFWwindow* getWindow(void)
	{
		return win;
	}

	void setMonitor(int idx)
	{
		monitor_idx = idx;
		getMonitorSize();
	}

	void getMonitorDimensions(int* width, int* height)
	{
		*width	= monitor_width;
		*height	= monitor_height;
	}

	__forceinline bool isKeyDown(int key)
	{
		return win && glfwGetKey(win, key) == GLFW_PRESS;
	}

	__forceinline bool shouldClose(void)
	{
		return win && glfwWindowShouldClose(win);
	}

	void swapBuffers(void)
	{
		if (win)
			glfwSwapBuffers(win);
	}

	void pollEvents(void)
	{
		glfwPollEvents();
	}

	void setKeyCallback(std::function<void(int, int, int, int)> callback)
	{
		key_callback_fn = callback;
	}

	void setCursorPosCallback(std::function<void(double, double)> callback)
	{
		cursor_pos_callback_fn = callback;
	}

	void setScrollCallback(std::function<void(double, double)> callback)
	{
		scroll_callback_fn = callback;
	}

	void setResizeCallback(std::function<void(int, int)> callback)
	{
		resize_callback_fn = callback;
	}
};
