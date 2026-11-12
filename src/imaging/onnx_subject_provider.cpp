#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "onnx_subject_provider.h"
#include "subject_input.h"
#define ORT_API_MANUAL_INIT
#include <onnxruntime_cxx_api.h>
#undef ORT_API_MANUAL_INIT
#include <windows.h>
#include <bcrypt.h>
#include <array>
#include <cmath>
#include <fstream>
#include <thread>
#include <chrono>
#include <mutex>
#include <algorithm>
namespace compositor::imaging {
namespace {
void initializeRuntime(){
    // Resolve once on the first inference request, before any Ort wrapper member
    // can dereference its API table. A system DLL may be older than our headers.
    static std::once_flag initialized;
    std::call_once(initialized,[]{
        const auto* base=OrtGetApiBase();const auto* api=base?base->GetApi(ORT_API_VERSION):nullptr;
        if(!api){
            std::array<wchar_t,32768> path{};const auto module=GetModuleHandleW(L"onnxruntime.dll");
            const DWORD length=module?GetModuleFileNameW(module,path.data(),DWORD(path.size())):0;
            const auto utf8=length?std::filesystem::path(std::wstring(path.data(),length)).u8string():std::u8string(u8"unknown module");
            const std::string loaded(utf8.begin(),utf8.end());
            throw std::runtime_error("ONNX Runtime API "+std::to_string(ORT_API_VERSION)+" is unavailable; loaded "+(base?base->GetVersionString():"unknown version")+" from "+loaded);
        }
        Ort::InitApi(api);
    });
}
std::string hash(const std::filesystem::path& file){
    BCRYPT_ALG_HANDLE alg{};BCRYPT_HASH_HANDLE value{};
    if(BCryptOpenAlgorithmProvider(&alg,BCRYPT_SHA256_ALGORITHM,nullptr,0)<0)throw std::runtime_error("SHA256 initialization failed");
    struct Clean{BCRYPT_ALG_HANDLE& a;BCRYPT_HASH_HANDLE& h;~Clean(){if(h)BCryptDestroyHash(h);if(a)BCryptCloseAlgorithmProvider(a,0);}}clean{alg,value};
    if(BCryptCreateHash(alg,&value,nullptr,0,nullptr,0,0)<0)throw std::runtime_error("SHA256 creation failed");
    std::ifstream stream(file,std::ios::binary);if(!stream)throw std::runtime_error("Local foreground model is missing");std::array<char,65536> buffer{};
    while(stream){stream.read(buffer.data(),buffer.size());if(BCryptHashData(value,reinterpret_cast<PUCHAR>(buffer.data()),ULONG(stream.gcount()),0)<0)throw std::runtime_error("Model hash failed");}
    if(!stream.eof())throw std::runtime_error("Model read failed");std::array<UCHAR,32> digest{};if(BCryptFinishHash(value,digest.data(),ULONG(digest.size()),0)<0)throw std::runtime_error("Model hash finalization failed");
    std::string out;const char* hex="0123456789abcdef";for(auto c:digest){out+=hex[c>>4];out+=hex[c&15];}return out;
}
// The pinned RGB input recipe is implemented by the bounded antialiased sampler.
std::vector<float> preprocess(const RgbaImage& image,const ImportOptions& options){
    return subjectInputTensor(image,options);
}
}
struct OnnxSubjectProvider::Impl {
    Ort::Env environment{ORT_LOGGING_LEVEL_WARNING,"CompositorForeground"};Ort::SessionOptions options;Ort::Session session{nullptr};std::mutex mutex;
    explicit Impl(const std::filesystem::path& path){
        if(hash(path)!=OnnxSubjectProvider::modelSha256)throw std::runtime_error("Foreground model hash does not match pinned conversion");
        environment.DisableTelemetryEvents();options.SetIntraOpNumThreads(8);options.SetInterOpNumThreads(1);options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        // Do not retain the model's multi-gigabyte temporary arena after an edit.
        // This reduces retained memory; a hard inference-process budget remains a separate gate.
        options.DisableCpuMemArena();options.DisableMemPattern();
        // No accelerator/provider registration: ONNX Runtime's mandatory CPU provider is used.
        session=Ort::Session(environment,path.c_str(),options);
        if(session.GetInputCount()!=1||session.GetOutputCount()!=1)throw std::runtime_error("Unexpected model graph interface");
        auto inType=session.GetInputTypeInfo(0);auto outType=session.GetOutputTypeInfo(0);auto in=inType.GetTensorTypeAndShapeInfo(),out=outType.GetTensorTypeAndShapeInfo();
        if(in.GetElementType()!=ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT||out.GetElementType()!=ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT||in.GetShape()!=std::vector<int64_t>{1,3,1024,1024}||out.GetShape()!=std::vector<int64_t>{1,1,1024,1024})throw std::runtime_error("Unexpected model tensor format");
    }
};
OnnxSubjectProvider::OnnxSubjectProvider(const std::filesystem::path& path){initializeRuntime();impl_=std::make_unique<Impl>(path);}
OnnxSubjectProvider::~OnnxSubjectProvider()=default;
GrayMask OnnxSubjectProvider::infer(const RgbaImage& image,const ImportOptions& options){
    return inferImpl(image,options,true);
}
void OnnxSubjectProvider::healthCheck(const ImportOptions& options){
    RgbaImage image{2,2,8,std::vector<std::uint8_t>(16,255)};
    validate(inferImpl(image,options,false));
}
GrayMask OnnxSubjectProvider::inferImpl(const RgbaImage& image,const ImportOptions& options,bool requireSubject){
    auto data=preprocess(image,options);std::lock_guard lock(impl_->mutex);
    auto memory=Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);const int64_t dims[]={1,3,1024,1024};auto tensor=Ort::Value::CreateTensor<float>(memory,data.data(),data.size(),dims,4);const char* names[]={"image"};const char* outputs[]={"mask"};Ort::RunOptions run;
    std::jthread cancellation([&](std::stop_token stop){while(!stop.stop_requested()){if(options.cancelled&&options.cancelled()){run.SetTerminate();return;}std::this_thread::sleep_for(std::chrono::milliseconds(20));}});
    auto result=impl_->session.Run(run,names,&tensor,1,outputs,1);cancellation.request_stop();checkCancelled(options);
    const auto outputInfo=result[0].GetTensorTypeAndShapeInfo();
    if(outputInfo.GetElementType()!=ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT||outputInfo.GetShape()!=std::vector<int64_t>{1,1,1024,1024})throw std::runtime_error("Foreground inference returned an unexpected output shape");
    const auto* mask=result[0].GetTensorData<float>();float max=0;for(std::size_t i=0;i<1024*1024;++i){if(!std::isfinite(mask[i]))throw std::runtime_error("Foreground inference returned non-finite mask");max=std::max(max,mask[i]);}if(requireSubject&&max<.5F)throw NoForegroundSubject();
    GrayMask out{image.width,image.height,image.width,std::vector<std::uint8_t>(std::size_t(image.width)*image.height)};
    for(std::uint32_t y=0;y<image.height;++y){checkCancelled(options);double sy=std::clamp((y+.5)*1024/image.height-.5,0.,1023.);int y0=int(sy),y1=std::min(y0+1,1023);float fy=float(sy-y0);for(std::uint32_t x=0;x<image.width;++x){double sx=std::clamp((x+.5)*1024/image.width-.5,0.,1023.);int x0=int(sx),x1=std::min(x0+1,1023);float fx=float(sx-x0);float value=(mask[y0*1024+x0]*(1-fx)+mask[y0*1024+x1]*fx)*(1-fy)+(mask[y1*1024+x0]*(1-fx)+mask[y1*1024+x1]*fx)*fy;out.pixels[std::size_t(y)*image.width+x]=std::uint8_t(std::clamp(value*255+.5F,0.F,255.F));}}return out;
}

std::vector<float> inferDepthMini(const std::filesystem::path& path,const RgbaImage& image,const ImportOptions& options){
    validate(image);
    if(image.width>2048||image.height>2048)throw std::runtime_error("Depth preview is too large");
    initializeRuntime();
    struct Model {
        Ort::Env environment{ORT_LOGGING_LEVEL_WARNING,"CompositorDepthMini"};
        Ort::Session session{nullptr};
        explicit Model(const std::filesystem::path& file){
            if(hash(file)!="fc6103d5245555696b4dcdc2fcbd5d52bad801fa26fc40bfc74f71b665e0c3e7")
                throw std::runtime_error("DepthMini model checksum mismatch");
            environment.DisableTelemetryEvents();
            Ort::SessionOptions settings;
            settings.SetIntraOpNumThreads(4);settings.SetInterOpNumThreads(1);
            settings.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
            session=Ort::Session(environment,file.c_str(),settings);
        }
    };
    static std::mutex mutex;
    static std::shared_ptr<Model> loaded;
    static std::filesystem::path loadedPath;
    std::shared_ptr<Model> model;
    {
        std::lock_guard lock(mutex);
        if(!loaded||loadedPath!=path){loaded=std::make_shared<Model>(path);loadedPath=path;}
        model=loaded;
    }
    constexpr std::array shapes{std::pair{512,512},std::pair{512,384},std::pair{512,288},std::pair{384,512},std::pair{288,512}};
    const double aspect=std::log(double(image.width)/image.height);
    const auto [w,h]=*std::min_element(shapes.begin(),shapes.end(),[aspect](auto a,auto b){
        return std::abs(std::log(double(a.first)/a.second)-aspect)<std::abs(std::log(double(b.first)/b.second)-aspect);
    });
    const auto bilinear=[](int width,int height,double x,double y,const auto& value){
        x=std::clamp(x,0.,double(width-1));y=std::clamp(y,0.,double(height-1));
        const int x0=int(x),y0=int(y),x1=std::min(x0+1,width-1),y1=std::min(y0+1,height-1);
        const auto a=std::lerp(value(x0,y0),value(x1,y0),float(x-x0));
        const auto b=std::lerp(value(x0,y1),value(x1,y1),float(x-x0));
        return std::lerp(a,b,float(y-y0));
    };
    const auto plane=std::size_t(w)*h;
    std::vector<float> input(plane*3);
    for(int y=0;y<h;++y){
        checkCancelled(options);
        for(int x=0;x<w;++x)for(int c=0;c<3;++c){
            input[plane*c+std::size_t(y)*w+x]=bilinear(int(image.width),int(image.height),(x+.5)*image.width/w-.5,(y+.5)*image.height/h-.5,
                [&](int sx,int sy){return float(image.pixels[std::size_t(sy)*image.stride+std::size_t(sx)*4+c]);});
        }
    }
    auto memory=Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);
    const int64_t shape[]{1,3,h,w};
    auto tensor=Ort::Value::CreateTensor<float>(memory,input.data(),input.size(),shape,4);
    const char* inputs[]{"image"};const char* outputs[]{"depth"};
    Ort::RunOptions run;
    std::jthread cancellation([&](std::stop_token stop){
        while(!stop.stop_requested()){
            if(options.cancelled&&options.cancelled()){run.SetTerminate();return;}
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    });
    auto result=model->session.Run(run,inputs,&tensor,1,outputs,1);
    cancellation.request_stop();checkCancelled(options);
    if(result[0].GetTensorTypeAndShapeInfo().GetShape()!=std::vector<int64_t>{1,1,h,w})
        throw std::runtime_error("DepthMini returned an unexpected image size");
    const auto* data=result[0].GetTensorData<float>();
    if(!std::all_of(data,data+plane,[](float v){return std::isfinite(v);}))throw std::runtime_error("DepthMini returned invalid depths");
    const auto [minimum,maximum]=std::minmax_element(data,data+plane);
    if(*maximum<=*minimum)return {};
    std::vector<float> depth(std::size_t(image.width)*image.height),guide(depth.size());
    for(unsigned y=0;y<image.height;++y){
        checkCancelled(options);
        for(unsigned x=0;x<image.width;++x){
            const auto i=std::size_t(y)*image.width+x;
            const auto* pixel=image.pixels.data()+std::size_t(y)*image.stride+std::size_t(x)*4;
            const float prediction=bilinear(w,h,(x+.5)*w/image.width-.5,(y+.5)*h/image.height-.5,
                [&](int sx,int sy){return data[std::size_t(sy)*w+sx];});
            depth[i]=(prediction-*minimum)/(*maximum-*minimum);
            guide[i]=(.2126f*pixel[0]+.7152f*pixel[1]+.0722f*pixel[2])/255;
        }
    }
    depth=guidedFilter(depth,guide,image.width,image.height,std::max(2,int(std::max(image.width,image.height)/48)),1e-3f);
    for(auto& value:depth)value=std::clamp(value,0.f,1.f);
    return depth;
}
}
