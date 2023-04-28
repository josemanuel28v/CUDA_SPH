#version 430

uniform mat4 view;
uniform mat4 proj;
uniform float radius;

in vec4 ppos;
in vec4 pcolor;
in vec4 vpos;
in vec2 vtextcoord;

out vec2 ftextcoord;
flat out vec4 center; 
flat out mat4 fmvp;
out vec4 fpos;
out vec4 fcolor;

void main()
{
    ftextcoord = vtextcoord;
    
    mat4 scale = mat4(0.0f);
    float diam = radius * 2.1; // para ocultar los bordes de la textura se añade 0.1 al factor 2
    scale[0][0] = diam;
    scale[1][1] = diam;
    scale[2][2] = diam;
    scale[3][3] = 1.0f;

    mat4 model = view;
    model[0][3] = ppos.x;
	model[1][3] = ppos.y;
	model[2][3] = ppos.z;
	model[3] = vec4(0.0f, 0.0f, 0.0f, 1.0f);
    model = scale * model;

	model = transpose(model);

    center = ppos;
    fcolor = pcolor;
    fpos = model * vpos;
    fmvp = proj * view * model;
    gl_Position = fmvp * vpos;
}