#version 430

uniform mat4 view;
uniform mat4 proj;
uniform float radius;

in vec4 ppos;
in vec4 vpos;
in vec2 vtextcoord;

out vec2 ftextcoord;
flat out vec4 center; 
flat out mat4 fmvp;
out vec4 fpos;

void main()
{
    ftextcoord = vtextcoord;
    
    mat4 scale = mat4(0.0f);
    float rad = radius * (2.0 + 0.1);
    scale[0][0] = rad;
    scale[1][1] = rad;
    scale[2][2] = rad;
    scale[3][3] = 1.0f;

    mat4 model = view;
    model[0][3] = ppos.x;
	model[1][3] = ppos.y;
	model[2][3] = ppos.z;
	model[3] = vec4(0.0f, 0.0f, 0.0f, 1.0f);
    model = scale * model;

	model = transpose(model);

    center = ppos;
    fpos = model * vpos;
    fmvp = proj * view * model;
    gl_Position = fmvp * vpos;
}