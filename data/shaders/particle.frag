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

vec3 computeNormalAndDepth()
{
    vec2 mapping = ftextcoord * 2.0f - 1.0f;
    float d = dot(mapping, mapping);

    if (d > 0.9) discard; // Descartar el exterior del circulo

    float z = sqrt(1.0f - d);
    mat3 normalMatrix = transpose(inverse(mat3(fmv)));
    vec3 normal = normalize(normalMatrix * vec3(mapping, z));

    // Corrección de perspectiva
    vec4 clipPos = proj * (fmv * center + radius * vec4(normal, 1.0));
    float depth = clipPos.z / clipPos.w;

    // Proyección NDC
    gl_FragDepth = ((gl_DepthRange.diff * depth) + gl_DepthRange.near + gl_DepthRange.far) / 2.0;

    return normal;
}

int discardExterior()
{
    vec2 mapping = ftextcoord * 2.0f - 1.0f;
    float d = dot(mapping, mapping);

    if (d > 0.9) 
        return 1;
    else 
        return 0;
}

void main()
{
    vec3 normal = computeNormalAndDepth();
    //if (discardExterior() == 1) discard;

    fragColor = fcolor;

    if (texturing == 1)
    {
        fragColor = texture2D(colorText, ftextcoord) * fragColor;
    }
}        
