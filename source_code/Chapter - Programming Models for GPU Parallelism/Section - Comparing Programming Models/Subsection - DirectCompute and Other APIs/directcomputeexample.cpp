#include <d3d11.h>
#include <d3dcompiler.h>
#include <vector>
#include <iostream>
#include <wrl/client.h>
#include <stdexcept>
#include <cstring>

#pragma comment(lib,"d3d11.lib")
#pragma comment(lib,"d3dcompiler.lib")

using Microsoft::WRL::ComPtr;

static const char* hlslSrc = R"(
StructuredBuffer<float> A : register(t0);
StructuredBuffer<float> B : register(t1);
RWStructuredBuffer<float> C : register(u0);

[numthreads(256,1,1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    C[tid.x] = A[tid.x] + B[tid.x];
}
)";

static constexpr UINT ELEMENT_COUNT = 1u << 20;
static constexpr UINT THREAD_GROUP_SIZE = 256;

void HR(HRESULT hr)
{
    if (FAILED(hr))
        throw std::runtime_error("D3D call failed");
}

int main()
try
{
    ComPtr<ID3D11Device> dev;
    ComPtr<ID3D11DeviceContext> ctx;
    D3D_FEATURE_LEVEL fl{};
    HR(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr,
                         D3D11_CREATE_DEVICE_SINGLETHREADED,
                         nullptr, 0, D3D11_SDK_VERSION, &dev, &fl, &ctx));

    ComPtr<ID3DBlob> csBlob;
    HR(D3DCompile(hlslSrc, std::strlen(hlslSrc), nullptr, nullptr, nullptr,
                  "main", "cs_5_0", 0, 0, &csBlob, nullptr));

    ComPtr<ID3D11ComputeShader> cs;
    HR(dev->CreateComputeShader(csBlob->GetBufferPointer(),
                                csBlob->GetBufferSize(), nullptr, &cs));

    std::vector<float> a(ELEMENT_COUNT, 1.0f);
    std::vector<float> b(ELEMENT_COUNT, 2.0f);

    D3D11_BUFFER_DESC bufDesc{};
    bufDesc.ByteWidth = ELEMENT_COUNT * sizeof(float);
    bufDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
    bufDesc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
    bufDesc.StructureByteStride = sizeof(float);

    auto createBuffer = [&](const void* init) -> ComPtr<ID3D11Buffer>
    {
        D3D11_SUBRESOURCE_DATA srd{ init, 0, 0 };
        ComPtr<ID3D11Buffer> buf;
        HR(dev->CreateBuffer(&bufDesc, init ? &srd : nullptr, &buf));
        return buf;
    };

    ComPtr<ID3D11Buffer> bufA = createBuffer(a.data());
    ComPtr<ID3D11Buffer> bufB = createBuffer(b.data());
    ComPtr<ID3D11Buffer> bufC = createBuffer(nullptr);

    D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc{};
    srvDesc.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
    srvDesc.Format = DXGI_FORMAT_UNKNOWN;
    srvDesc.Buffer.NumElements = ELEMENT_COUNT;

    ComPtr<ID3D11ShaderResourceView> srvA, srvB;
    HR(dev->CreateShaderResourceView(bufA.Get(), &srvDesc, &srvA));
    HR(dev->CreateShaderResourceView(bufB.Get(), &srvDesc, &srvB));

    D3D11_UNORDERED_ACCESS_VIEW_DESC uavDesc{};
    uavDesc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
    uavDesc.Format = DXGI_FORMAT_UNKNOWN;
    uavDesc.Buffer.NumElements = ELEMENT_COUNT;

    ComPtr<ID3D11UnorderedAccessView> uavC;
    HR(dev->CreateUnorderedAccessView(bufC.Get(), &uavDesc, &uavC));

    ctx->CSSetShader(cs.Get(), nullptr, 0);
    ID3D11ShaderResourceView* srvs[] = { srvA.Get(), srvB.Get() };
    ctx->CSSetShaderResources(0, 2, srvs);
    ID3D11UnorderedAccessView* uavs[] = { uavC.Get() };
    ctx->CSSetUnorderedAccessViews(0, 1, uavs, nullptr);

    ctx->Dispatch((ELEMENT_COUNT + THREAD_GROUP_SIZE - 1) / THREAD_GROUP_SIZE, 1, 1);

    D3D11_BUFFER_DESC readDesc{};
    readDesc.ByteWidth = bufDesc.ByteWidth;
    readDesc.Usage = D3D11_USAGE_STAGING;
    readDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    readDesc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
    readDesc.StructureByteStride = sizeof(float);

    ComPtr<ID3D11Buffer> staging;
    HR(dev->CreateBuffer(&readDesc, nullptr, &staging));
    ctx->CopyResource(staging.Get(), bufC.Get());

    D3D11_MAPPED_SUBRESOURCE mapped{};
    HR(ctx->Map(staging.Get(), 0, D3D11_MAP_READ, 0, &mapped));
    const float* out = static_cast<const float*>(mapped.pData);

    for (size_t i = 0; i < 10; ++i)
        std::cout << out[i] << ' ';
    std::cout << '\n';

    ctx->Unmap(staging.Get(), 0);
    return 0;
}
catch (const std::exception& e)
{
    std::cerr << e.what() << '\n';
    return -1;
}