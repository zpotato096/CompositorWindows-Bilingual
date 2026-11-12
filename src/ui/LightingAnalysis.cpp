#include "LightingAnalysis.h"
#include "imaging/onnx_subject_provider.h"
#include <QDir>
#include <algorithm>
#include <cmath>
#include <mutex>

namespace compositor::ui {

LightingAnalysis prepareLightingAnalysis(std::shared_ptr<const Raster> source,const QString& modelDirectory,bool wantsDepth){
    struct Cache {
        std::mutex mutex;
        std::shared_ptr<const filters::LightingScene> scene;
    };
    return [source=std::move(source),modelDirectory,wantsDepth,cache=std::make_shared<Cache>()](const std::function<bool()>& cancel){
        std::lock_guard lock(cache->mutex);
        if(cache->scene)return cache->scene;
        imaging::ImportOptions options;options.cancelled=cancel;
        imaging::checkCancelled(options);
        const double scale=std::min(1.,768./std::max(source->width,source->height));
        const int width=std::max(1,int(std::lround(source->width*scale))),height=std::max(1,int(std::lround(source->height*scale)));
        const auto stride=std::size_t(width)*4;
        imaging::RgbaImage image{unsigned(width),unsigned(height),stride,std::vector<uint8_t>(stride*height)};
        const auto interpolate=[](Pixel left,Pixel right,double factor){
            auto c=[&](auto a,auto b){return uint8_t(std::lround(std::lerp(double(a),double(b),factor)));};
            return Pixel{c(left.r,right.r),c(left.g,right.g),c(left.b,right.b),c(left.a,right.a)};
        };
        std::size_t transparent=0,opaque=0;
        for(int y=0;y<height;++y){
            imaging::checkCancelled(options);
            const auto sy=std::clamp((y+.5)*source->height/height-.5,0.,double(source->height-1));
            const int top=int(sy),bottom=std::min(top+1,source->height-1);
            for(int x=0;x<width;++x){
                const auto sx=std::clamp((x+.5)*source->width/width-.5,0.,double(source->width-1));
                const int left=int(sx),right=std::min(left+1,source->width-1);
                const auto pixel=interpolate(interpolate(source->pixel(left,top),source->pixel(right,top),sx-left),
                    interpolate(source->pixel(left,bottom),source->pixel(right,bottom),sx-left),sy-top);
                const auto offset=std::size_t(y)*stride+std::size_t(x)*4;
                image.pixels[offset]=pixel.r;image.pixels[offset+1]=pixel.g;image.pixels[offset+2]=pixel.b;image.pixels[offset+3]=pixel.a;
                transparent+=pixel.a<24;opaque+=pixel.a>96;
            }
        }
        auto scene=std::make_shared<filters::LightingScene>();
        scene->sample=Raster::fromRgba(width,height,image.pixels.data(),stride,cancel);
        const auto pixels=std::size_t(width)*height;
        if(transparent>pixels/20&&opaque>pixels/100){
            scene->subject.resize(pixels);
            for(std::size_t i=0;i<pixels;++i)scene->subject[i]=image.pixels[i*4+3]/255.f;
        }else{
            try{
                const auto file=QDir(modelDirectory).filePath("birefnet-lite.onnx");
                imaging::OnnxSubjectProvider model(std::filesystem::path(file.toStdWString()));
                auto mask=model.infer(image,options);
                mask=imaging::refineSubjectMask(mask,image,{true,std::max(2.,std::max(width,height)*.008),0,0},false,nullptr,options);
                scene->subject.resize(pixels);
                for(std::size_t i=0;i<pixels;++i)scene->subject[i]=mask.pixels[i]/255.f;
            }catch(const imaging::NoForegroundSubject&){}
            catch(const std::exception&){if(!wantsDepth)throw;imaging::checkCancelled(options);}
        }
        if(wantsDepth){
            try{
                const auto file=QDir(modelDirectory).filePath("depthmini.onnx");
                scene->depth=imaging::inferDepthMini(std::filesystem::path(file.toStdWString()),image,options);
            }catch(const std::exception& error){
                imaging::checkCancelled(options);scene->depthIssue=error.what();
            }
        }
        imaging::checkCancelled(options);
        cache->scene=std::move(scene);
        return cache->scene;
    };
}

}
