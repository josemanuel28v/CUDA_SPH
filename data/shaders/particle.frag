#version 430

uniform sampler2D colorText;
uniform int texturing;
uniform float radius;
uniform mat4 proj;

in vec2 ftextcoord;
flat in vec4 center; 
flat in mat4 fmv;
in vec4 fcolor;

out vec4 fragColor;

void computeNormalAndDepth(out vec3 normal)
{
    vec2 mapping = ftextcoord * 2.0f - 1.0f;
    float d = dot(mapping, mapping);

    if (d > 1.0) discard; // Descartar el exterior del circulo

    float z = sqrt(1.0f - d);
    normal = /*mat3(inverse(transpose(fmv))) **/ normalize(vec3(mapping, z));
    vec3 cameraPos = vec3(fmv * center) + radius * normal;

    vec4 clipPos = proj * vec4(cameraPos, 1.0);
    float ndcDepth = clipPos.z / clipPos.w;
    gl_FragDepth = ((gl_DepthRange.diff * ndcDepth) + gl_DepthRange.near + gl_DepthRange.far) / 2.0;
}

void main()
{
    vec3 normal;
    computeNormalAndDepth(normal);

    fragColor = fcolor;

    if (texturing == 1)
    {
        fragColor = texture2D(colorText, ftextcoord) * fragColor;
    }
}        
