#version 430

uniform sampler2D colorText;
uniform int texturing;
uniform float radius;
uniform mat4 proj;

in vec2 ftextcoord;
flat in vec4 center; 
flat in mat4 fmvp;
in vec4 fpos;

out vec4 fragColor;

// void makeSphereA(out vec3 cameraPos, out vec3 cameraNormal)
// {
// 	vec2 mapping = ftextcoord * 2.0F - 1.0F;
//     //float d = dot(mapping, mapping);

//     vec3 cameraSpherePos = vec3(fmvp * fpos);

//     vec3 cameraPlanePos = vec3(mapping * radius, 0.0) + cameraSpherePos;
//     vec3 rayDirection = normalize(cameraPlanePos);

//     float B = 2.0 * dot(rayDirection, -cameraSpherePos);
//     float C = dot(cameraSpherePos, cameraSpherePos) - (radius * radius);

//     float det = (B * B) - (4 * C);
//     if(det < 0.0)
//         discard;

//     float sqrtDet = sqrt(det);
//     float posT = (-B + sqrtDet)/2;
//     float negT = (-B - sqrtDet)/2;

//     float intersectT = min(posT, negT);
//     cameraPos = rayDirection * intersectT;
//     cameraNormal = normalize(cameraPos - cameraSpherePos);

// 	vec4 clipPos = proj * vec4(cameraPos, 1.0);
//     float ndcDepth = clipPos.z / clipPos.w;
//     gl_FragDepth = ((gl_DepthRange.diff * ndcDepth) + gl_DepthRange.near + gl_DepthRange.far) / 2.0;

//     //cameraNormal = mat3(transpose(fmvp)) * cameraNormal;
// 	//cameraNormal = mat3(inverse(transpose(fmvp))) * cameraNormal;
// 	cameraNormal = normalize(cameraNormal);
// }

void main()
{
    if (length(center - fpos) > radius) discard;
    //vec3 cameraPos, normal;
    //makeSphere(cameraPos, normal);

    fragColor = vec4(1.0f);

    if (texturing == 1)
    {
        fragColor = texture2D(colorText, ftextcoord);
    }
}        
