#version 430 core

uniform int u_maxIterations;
uniform int u_colorStyle;
uniform int u_baseColor;
uniform int u_colorArray[4];
uniform int u_colorCount;

in vec2 v_texCoord;

out vec4 FragColor;

const int STYLE_MONO = 0;
const int STYLE_MULTIPLE = 1;
const int STYLE_OPPOSITE = 2;
const int STYLE_CONTRASTED = 3;
const int STYLE_GRAPHIC = 4;
const int STYLE_ZEBRA = 5;
const int STYLE_TRIAD = 6;
const int STYLE_TETRA = 7;

uniform sampler2D u_iterationTexture;

vec4 intToVec4(int color)
{
    float r = float((color >> 16) & 0xFF) / 255.0;
    float g = float((color >> 8) & 0xFF) / 255.0;
    float b = float(color & 0xFF) / 255.0;
    float a = float((color >> 24) & 0xFF) / 255.0;
    return vec4(r, g, b, a);
}

vec4 interpolate(int startcolor, int endcolor, float f)
{
    vec3 start_rgb;
    vec3 end_rgb;

    start_rgb.r = float((startcolor >> 16) & 0xFF);
    start_rgb.g = float((startcolor >> 8) & 0xFF);
    start_rgb.b = float((startcolor >> 0) & 0xFF);

    end_rgb.r = float((endcolor >> 16) & 0xFF);
    end_rgb.g = float((endcolor >> 8) & 0xFF);
    end_rgb.b = float((endcolor >> 0) & 0xFF);

    vec3 result;
    result.r = (end_rgb.r - start_rgb.r) * f + start_rgb.r;
    result.g = (end_rgb.g - start_rgb.g) * f + start_rgb.g;
    result.b = (end_rgb.b - start_rgb.b) * f + start_rgb.b;

    return vec4(result / 255.0, 1.0);
}

vec4 getColorMono(int iterations)
{
    if (iterations == u_maxIterations - 1)
        return vec4(0.0, 0.0, 0.0, 1.0);

    int col1 = 0x000000;
    int col2 = u_baseColor;

    if (iterations >= u_maxIterations / 2)
    {
        col1 = u_baseColor;
        col2 = 0xFFFFFF;
        iterations -= u_maxIterations / 2;
    }

    float f = float(iterations) / float(u_maxIterations / 2);
    return interpolate(col1, col2, f);
}

vec4 getColorMultiple(int iterations)
{
    if (iterations == u_maxIterations - 1)
        return vec4(0.0, 0.0, 0.0, 1.0);

    int segmentSize = u_maxIterations / (u_colorCount - 1);
    int segmentIndex = iterations / segmentSize;

    segmentIndex = min(segmentIndex, u_colorCount - 2);

    int startColor = u_colorArray[segmentIndex];
    int endColor = u_colorArray[segmentIndex + 1];

    float f = float(iterations % segmentSize) / float(segmentSize);
    return interpolate(startColor, endColor, f);
}

vec4 getColorOpposites(int iterations)
{
    if (iterations == u_maxIterations - 1)
        return vec4(0.0, 0.0, 0.0, 1.0);

    int r = (u_baseColor >> 16) & 0xFF;
    int g = (u_baseColor >> 8) & 0xFF;
    int b = (u_baseColor >> 0) & 0xFF;

    r += iterations % 255;
    g += iterations % 255;
    b += iterations % 255;

    return vec4(float(r) / 255.0, float(g) / 255.0, float(b) / 255.0, 1.0);
}

vec4 getColorContrasted(int iterations)
{
    if (iterations == u_maxIterations - 1)
        return vec4(0.0, 0.0, 0.0, 1.0);

    int r = (u_baseColor >> 16) & 0xFF;
    int g = (u_baseColor >> 8) & 0xFF;
    int b = (u_baseColor >> 0) & 0xFF;

    if (r != 255) r += iterations % 255;
    if (g != 255) g += iterations % 255;
    if (b != 255) b += iterations % 255;

    r = min(r, 255);
    g = min(g, 255);
    b = min(b, 255);

    return vec4(float(r) / 255.0, float(g) / 255.0, float(b) / 255.0, 1.0);
}

vec4 getColorGraphic(int iterations)
{
    if (iterations == u_maxIterations - 1)
        return vec4(0.0, 0.0, 0.0, 1.0);

    int r = (u_baseColor >> 16) & 0xFF;
    int g = (u_baseColor >> 8) & 0xFF;
    int b = (u_baseColor >> 0) & 0xFF;

    while (r < 0x33 || g < 0x33 || b < 0x33)
    {
        if (r != 255) r++;
        if (g != 255) g++;
        if (b != 255) b++;
    }

    r -= iterations % 255;
    g -= iterations % 255;
    b -= iterations % 255;

    r = max(0, r);
    g = max(0, g);
    b = max(0, b);

    return vec4(float(r) / 255.0, float(g) / 255.0, float(b) / 255.0, 1.0);
}

vec4 getPercentColor(int color, float percent)
{
    float percentage = (percent / 100.0) * 256.0;

    int r = (color >> 16) & 0xFF;
    int g = (color >> 8) & 0xFF;
    int b = (color >> 0) & 0xFF;

    int tr = int(float(r) + percentage) - 256;
    int tg = int(float(g) + percentage) - 256;
    int tb = int(float(b) + percentage) - 256;

    tr = clamp(tr, 0, 255);
    tg = clamp(tg, 0, 255);
    tb = clamp(tb, 0, 255);

    return vec4(float(tr) / 255.0, float(tg) / 255.0, float(tb) / 255.0, 1.0);
}

vec4 getColorZebra(int iterations)
{
    if (iterations == u_maxIterations - 1)
        return vec4(0.0, 0.0, 0.0, 1.0);

    if (iterations % 2 == 0)
        return intToVec4(u_baseColor);
    else
        return getPercentColor(u_baseColor, 50.0);
}

vec4 getColorTriad(int iterations)
{
    if (iterations == u_maxIterations - 1)
        return vec4(0.0, 0.0, 0.0, 1.0);

    int remainder = iterations % 3;

    if (remainder == 0)
        return intToVec4(u_baseColor);
    else if (remainder == 1)
        return getPercentColor(u_baseColor, 33.0);
    else
        return getPercentColor(u_baseColor, 66.0);
}

vec4 getColorTetra(int iterations)
{
    if (iterations == u_maxIterations - 1)
        return vec4(0.0, 0.0, 0.0, 1.0);

    int remainder = iterations % 4;

    if (remainder == 0)
        return intToVec4(u_baseColor);
    else if (remainder == 1)
        return getPercentColor(u_baseColor, 25.0);
    else if (remainder == 2)
        return getPercentColor(u_baseColor, 50.0);
    else
        return getPercentColor(u_baseColor, 75.0);
}

void main()
{
    int iterations = int(texture(u_iterationTexture, v_texCoord).r);

    if (u_colorStyle == STYLE_MONO)
        FragColor = getColorMono(iterations);
    else if (u_colorStyle == STYLE_MULTIPLE)
        FragColor = getColorMultiple(iterations);
    else if (u_colorStyle == STYLE_OPPOSITE)
        FragColor = getColorOpposites(iterations);
    else if (u_colorStyle == STYLE_CONTRASTED)
        FragColor = getColorContrasted(iterations);
    else if (u_colorStyle == STYLE_GRAPHIC)
        FragColor = getColorGraphic(iterations);
    else if (u_colorStyle == STYLE_ZEBRA)
        FragColor = getColorZebra(iterations);
    else if (u_colorStyle == STYLE_TRIAD)
        FragColor = getColorTriad(iterations);
    else if (u_colorStyle == STYLE_TETRA)
        FragColor = getColorTetra(iterations);
    else
        FragColor = vec4(1.0, 0.0, 1.0, 1.0);
}
