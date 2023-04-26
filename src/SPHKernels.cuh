#include <glm/glm.hpp>
#include "types.h"

__device__ float cubicW(glm::vec3 rij, float h, float k)
{
    float r = length(rij);
    float q = r / h;

    if (q <= 1.0)
    {
        if (q <= 0.5)
        {
            float q2 = q * q;
            float q3 = q2 * q;
            return k * (6.0f * q3 - 6.0f * q2 + 1.0f);
        }
        else
        {
            return k * (2.0f * pow(1.0f - q, 3.0f));
        }
    }

    return 0.0f;
}

__device__ glm::vec3 cubicWgrad(glm::vec3 rij, float h) // No se usa (Aparentemente la fuerza de presion se comporta mejor con el spiky kernel)
{
    float rl = length(rij);
    float q = rl / h;
    float l = 48.0f / (M_PI * h * h * h);

    if ((rl > 1.0e-5) && (q <= 1.0))
    {
        glm::vec3 gradq = rij * (1.0f / (rl * h));
        if (q <= 0.5)
        {
            return l * q * (3.0f*q - 2.0f) * gradq;
        }
        else
        {
            float factor = 1.0f - q;
            return l * (-factor * factor) * gradq;
        }
    }

    return glm::vec3(0.0f);
}

__device__ float poly6W(glm::vec3 r, float h2, float poly6Const) // No se usa (No sirve si h se calcula como 4 * prad)
{
    float modr2 = dot(r, r);

    if (modr2 < h2) 
    {
        float aux = h2 - modr2;
        return poly6Const * aux * aux * aux;
    }
    else 
    {
        return 0.0f;
    }
}

__device__ glm::vec3 spikyW(glm::vec3 r, float h, float spikyConst)
{
    float modr = length(r);

    if (modr > 0 && modr < h)
    {
        float aux = h - modr;
        return - spikyConst * aux * aux * r / modr; 
    }

    return glm::vec3(0.0f);
}

__device__ float laplW(glm::vec3 r, float h, float laplConst)
{
    float modr = length(r);

    if (modr < h)
    {
        return laplConst * (h - modr);
    }

    return 0;
}