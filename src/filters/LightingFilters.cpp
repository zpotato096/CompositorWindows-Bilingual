// RimLight.swift and VolumetricLight.swift from Compositor v1.6.6.
// Copyright (c) 2026 Wonder Assembly LLC. MIT notice retained.
#include "LightingFilters.h"
#include "PixelFilters.h"
#include "graphics/ParallelBatch.h"
#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>

namespace compositor::filters {
namespace {
using RGB=std::array<float,3>;
using Cancel=std::function<bool()>;

void check(const Cancel& cancel){if(cancel&&cancel())throw std::runtime_error("Filter cancelled");}
float unit(double value){return std::isfinite(value)?float(std::clamp(value,0.,1.)):0.f;}
float control(double value,double fallback=0){return unit((std::isfinite(value)?value:fallback)/100);}
float smooth(float lo,float hi,float value){const auto t=unit((value-lo)/(hi-lo));return t*t*(3-2*t);}
float luma(RGB pixel){return .2126f*pixel[0]+.7152f*pixel[1]+.0722f*pixel[2];}

struct Plane {
    int width,height;
    std::vector<float> values;
    Plane(int w,int h):width(w),height(h),values(std::size_t(w)*h){}
    float& operator[](std::size_t i){return values[i];}
    float operator[](std::size_t i)const{return values[i];}
    float at(int x,int y)const{return values[std::size_t(std::clamp(y,0,height-1))*width+std::clamp(x,0,width-1)];}
    float sample(float x,float y)const{
        x=std::clamp(x,0.f,float(width-1));y=std::clamp(y,0.f,float(height-1));
        const int left=int(x),top=int(y);
        return std::lerp(std::lerp(at(left,top),at(left+1,top),x-left),
                         std::lerp(at(left,top+1),at(left+1,top+1),x-left),y-top);
    }
};

void box(const Plane& input,Plane& output,int radius,bool horizontal,const Cancel& cancel){
    if(radius==0){output.values=input.values;return;}
    const int lines=horizontal?input.height:input.width,length=horizontal?input.width:input.height;
    graphics::runParallelBatch(std::size_t(lines),6,[&](std::size_t line){
        check(cancel);
        auto value=[&](int i){return horizontal?input.at(i,int(line)):input.at(int(line),i);};
        double sum=0;
        for(int i=-radius;i<=radius;++i)sum+=value(i);
        for(int i=0;i<length;++i){
            const auto index=horizontal?line*input.width+i:std::size_t(i)*input.width+line;
            output[index]=float(sum/(2*radius+1));
            sum+=value(i+radius+1)-value(i-radius);
        }
    });
}

Plane blur(const Plane& input,float sigma,const Cancel& cancel){
    if(sigma<.35f)return input;
    // Three successive box filters approximate a Gaussian in constant work per pixel.
    const float width=std::sqrt(4*sigma*sigma+1);
    const int narrow=std::max(1,int(width)|1)-(int(width)%2==0&&width>=2?2:0);
    const int wide=narrow+2;
    const int narrowPasses=std::clamp(int(std::lround((12*sigma*sigma-3*narrow*narrow-12*narrow-9)/(-4*narrow-4))),0,3);
    Plane current=input,scratch(input.width,input.height);
    for(int i=0;i<3;++i){
        const int radius=((i<narrowPasses?narrow:wide)-1)/2;
        box(current,scratch,radius,true,cancel);box(scratch,current,radius,false,cancel);
    }
    return current;
}

struct ImagePlanes {
    Plane red,green,blue,gray,outline,coverage;
    Point brightest;
    float peak{};
    explicit ImagePlanes(const LightingScene& scene,const Cancel& cancel)
        :red(scene.sample->width,scene.sample->height),green(red),blue(red),gray(red),outline(red),coverage(red){
        for(int y=0;y<red.height;++y){
            check(cancel);
            for(int x=0;x<red.width;++x){
                const auto i=std::size_t(y)*red.width+x;
                const auto p=scene.sample->pixel(x,y);
                const auto a=float(p.a);
                if(a){red[i]=float(srgbToLinear(p.r/a));green[i]=float(srgbToLinear(p.g/a));blue[i]=float(srgbToLinear(p.b/a));}
                gray[i]=luma({red[i],green[i],blue[i]})*a/255;
                if(gray[i]>peak){peak=gray[i];brightest={(x+.5)/red.width,(y+.5)/red.height};}
                coverage[i]=scene.subject.empty()?0:unit(scene.subject[i]);
                outline[i]=unit(coverage[i]*1.6-.3);
            }
        }
        if(peak==0)brightest={.5,.5};
    }
};

std::shared_ptr<const LightingScene> getScene(const Request& request){
    if(!request.prepareLighting)throw std::runtime_error("Lighting analysis is unavailable");
    auto scene=request.prepareLighting(request.limits.cancelled);
    if(!scene||!scene->sample||scene->sample->width<1||scene->sample->height<1||
       scene->sample->width>1024||scene->sample->height>1024)throw std::runtime_error("Invalid lighting analysis");
    const auto pixels=std::size_t(scene->sample->width)*scene->sample->height;
    for(const auto* map:{&scene->subject,&scene->depth})
        if(!map->empty()&&(map->size()!=pixels||!std::all_of(map->begin(),map->end(),[](float v){return std::isfinite(v); })))
            throw std::runtime_error("Invalid lighting mask or depth");
    return scene;
}

// Invert the error function to recover the distance to a Gaussian-blurred edge.
float inverseError(float x){
    float w=-std::log((1-x)*(1+x)),p;
    if(w<5){
        w-=2.5f;
        p=2.81022636e-8f;p=3.43273939e-7f+p*w;p=-3.5233877e-6f+p*w;p=-4.39150654e-6f+p*w;
        p=.00021858087f+p*w;p=-.00125372503f+p*w;p=-.00417768164f+p*w;p=.246640727f+p*w;p=1.50140941f+p*w;
    }else{
        w=std::sqrt(w)-3;
        p=-.000200214257f;p=.000100950558f+p*w;p=.00134934322f+p*w;p=-.00367342844f+p*w;
        p=.00573950773f+p*w;p=-.0076224613f+p*w;p=.00943887047f+p*w;p=1.00167406f+p*w;p=2.83297682f+p*w;
    }
    return p*x;
}

std::array<float,2> inward(const Plane& edge,int x,int y){
    const float dx=edge.at(x+1,y)-edge.at(x-1,y),dy=edge.at(x,y-1)-edge.at(x,y+1);
    const auto length=std::hypot(dx,dy);
    return length>1e-6f?std::array{dx/length,dy/length}:std::array{0.f,0.f};
}

std::array<Plane,3> rimField(const ImagePlanes& image,const RimLightingSettings& settings,const Cancel& cancel){
    const int w=image.gray.width,h=image.gray.height;
    const float side=float(std::max(w,h));
    const auto fine=blur(image.gray,1,cancel),coarse=blur(image.gray,std::max(2.f,side/400),cancel);
    const auto near=blur(image.outline,side*.015f,cancel);
    Plane open(w,h),weighted(w,h),squares(w,h);
    for(std::size_t i=0;i<open.values.size();++i){
        open[i]=1-image.outline[i];weighted[i]=image.gray[i]*open[i];squares[i]=image.gray[i]*weighted[i];
    }
    const auto backdropWeight=blur(open,side*.01f,cancel);
    const auto backdrop=blur(weighted,side*.01f,cancel),spread=blur(squares,side*.01f,cancel);
    std::array<Plane,3> field{Plane(w,h),Plane(w,h),Plane(w,h)};
    for(int lamp=0;lamp<std::clamp(settings.count,1,3);++lamp){
        const auto& light=settings.lights[lamp];
        const float strength=10*control(light.strength),choke=control(light.choke);
        if(strength==0)continue;
        const float width=side*(.002f+control(light.width)*.03f),reach=std::max(1.f,width/1.5f);
        const double angle=(std::isfinite(light.angle)?std::remainder(light.angle,360.):135.)*std::numbers::pi/180;
        const auto edge=blur(image.outline,reach,cancel);
        RGB tint{};for(int c=0;c<3;++c)tint[c]=strength*float(srgbToLinear(unit(light.color[c])));
        auto faces=[&](float x,float y){
            const auto dot=(x*std::cos(angle)+y*std::sin(angle)-.2)/(std::sqrt(1.04)*std::sqrt(x*x+y*y+1));
            return float(std::pow(std::max(0.,dot),1.6));
        };
        graphics::runParallelBatch(std::size_t(h),6,[&](std::size_t row){
            check(cancel);const int y=int(row);
            for(int x=0;x<w;++x){
                const auto i=row*w+x;
                if(image.coverage[i]<=.002f&&near[i]<=.002f)continue;
                const float depth=reach*std::numbers::sqrt2_v<float>*inverseError(2*std::clamp(edge[i],.001f,.999f)-1);
                const auto normal=inward(edge,x,y);
                auto texture=[&](int px,int py){return fine.at(px,py)-coarse.at(px,py);};
                const float dx=(texture(x+1,y)-texture(x-1,y))*.5f,dy=(texture(x,y-1)-texture(x,y+1))*.5f;
                const float t=std::max(depth,0.f)/width;
                const float held=1-smooth(std::max(0.f,(choke-.3f)/.7f)*.95f,1,t);
                const float reaches=(choke<.3f?std::lerp(std::exp(-2.2f*t),held,choke/.3f):held)*(1-smooth(1.5f,1.9f,t));
                const float inside=smooth(-1.5f,1.5f,depth);
                const float lit=faces(-normal[0]*2.75f-dx*side/250,-normal[1]*2.75f-dy*side/250);
                const float mean=backdrop[i]/(backdropWeight[i]+.001f);
                const float variance=std::abs(spread[i]/(backdropWeight[i]+.001f)-mean*mean);
                const float strand=unit((std::abs(image.gray[i]-mean)-.03f-2.5f*std::sqrt(variance))/.1f)*smooth(0,.1f,near[i])*(1-inside);
                const auto out=inward(near,x,y);
                const float amount=lit*reaches*std::clamp(1+texture(x,y)*6,.3f,1.8f)*image.coverage[i]*inside
                    +strand*faces(-out[0]*2.75f,-out[1]*2.75f);
                const RGB albedo{image.red[i],image.green[i],image.blue[i]};
                for(int c=0;c<3;++c)field[c][i]+=tint[c]*(albedo[c]*.7f+.45f)*amount;
            }
        });
    }
    if(const float glow=4*control(settings.glow);glow>0){
        const auto surrounding=blur(image.outline,side*.008f,cancel);
        for(auto& channel:field){
            const auto halo=blur(channel,side*.012f,cancel),haze=blur(channel,side*.035f,cancel);
            for(std::size_t i=0;i<channel.values.size();++i)channel[i]+=glow*(.6f*halo[i]+haze[i])*unit(2-2*surrounding[i]);
        }
    }
    return field;
}

Plane composedDepth(const LightingScene& scene,const ImagePlanes& image,const VolumetricLightingSettings& settings,const Cancel& cancel){
    Plane depth=image.gray;
    depth.values=scene.depth;
    if(depth.values.empty())return depth;
    const float contrast=control(settings.depthContrast);
    if(contrast>0){
        auto sorted=depth.values;std::sort(sorted.begin(),sorted.end());
        for(auto& value:depth.values)value=std::lerp(value,float(std::lower_bound(sorted.begin(),sorted.end(),value)-sorted.begin())/float(sorted.size()),contrast);
    }
    const float side=float(std::max(depth.width,depth.height));
    Plane open(depth.width,depth.height),background=open;
    for(std::size_t i=0;i<open.values.size();++i){open[i]=1-image.outline[i];background[i]=open[i]*depth[i];}
    const auto nearby=blur(open,side*.016f,cancel),far=blur(background,side*.016f,cancel);
    for(std::size_t i=0;i<depth.values.size();++i){
        const float original=depth[i];
        const float corrected=nearby[i]>.03f?far[i]/nearby[i]:original;
        const float backgroundDepth=std::lerp(original,corrected,smooth(.1f,.9f,1-nearby[i]));
        depth[i]=unit(std::lerp(backgroundDepth,settings.keepsSubjectClear?1.f:original,image.coverage[i]));
    }
    return depth;
}

struct VolumeField {
    Plane fog,beams;
    std::array<Plane,3> fogColor;
};
float hash(float n){const float t=std::sin(n*12.9898f)*43758.5453f;return t-std::floor(t);}
float around(float angle,float cells,float seed){
    const float u=(angle/(2*std::numbers::pi_v<float>)+.5f)*cells,i=std::floor(u);
    return std::lerp(hash(std::fmod(i,cells)+seed),hash(std::fmod(i+1,cells)+seed),smooth(0,1,u-i));
}

VolumeField volumeField(const LightingScene& scene,const ImagePlanes& image,const Request& request){
    const auto& settings=request.settings.volume;
    const auto& cancel=request.limits.cancelled;
    const int w=image.gray.width,h=image.gray.height;
    const float side=float(std::max(w,h));
    const auto position=settings.center&&std::isfinite(settings.center->x)&&std::isfinite(settings.center->y)
        ?request.transform.toUnit(*settings.center):image.brightest;
    const float lightX=float(std::clamp(position.x,-4.,5.)*w-.5),lightY=float(std::clamp(position.y,-4.,5.)*h-.5);
    const float size=std::max(1.f,side*(.02f+control(settings.size)*.4f));
    const float shafts=control(settings.shafts),length=control(settings.length),haze=control(settings.haze);
    const auto near=blur(image.outline,side*.008f,cancel);
    const auto depth=composedDepth(scene,image,settings,cancel);
    const float distance=control(settings.distance);
    float start=distance*.6f,end=start+.5f;
    if(!depth.values.empty()){
        auto distances=scene.depth;for(auto& value:distances)value=1-value;std::sort(distances.begin(),distances.end());
        auto at=[&](float fraction){return distances[std::size_t(fraction*float(distances.size()-1))];};
        start=std::min(at(distance),.9f);end=std::max(at(std::min(1.f,distance+.45f)),start+.45f);
    }
    VolumeField result{Plane(w,h),Plane(w,h),{Plane(w,h),Plane(w,h),Plane(w,h)}};
    Plane emission(w,h);
    graphics::runParallelBatch(std::size_t(h),6,[&](std::size_t row){
        check(cancel);
        for(int x=0;x<w;++x){
            const auto i=row*w+x;
            const float dx=x-lightX,dy=float(row)-lightY,d=std::hypot(dx,dy);
            const float bright=unit(image.gray[i]/std::max(.05f,image.peak));
            const float disc=std::exp(-2*(d/size)*(d/size));
            const float openings=unit((bright-.45f)/.45f)*std::exp(-d/(side*.5f));
            const float angle=std::atan2(dy,dx),wave=.6f*around(angle,97,3.1f)+.4f*around(angle,251,7.7f);
            emission[i]=(disc*(.25f+.75f*unit((bright-.08f)/.3f))+openings)*(1-image.outline[i])*(1-shafts+shafts*wave*wave*1.8f);
        }
    });
    std::array<float,96> weights{};
    for(std::size_t i=0;i<weights.size();++i)weights[i]=i?weights[i-1]*.985f:1;
    RGB tint{};for(int c=0;c<3;++c)tint[c]=float(srgbToLinear(unit(settings.color[c])));
    const std::array source{&image.red,&image.green,&image.blue};
    std::array<Plane,3> air{blur(*source[0],side*.10f,cancel),blur(*source[1],side*.10f,cancel),blur(*source[2],side*.10f,cancel)};
    std::array<Plane,3> glows{Plane(w,h),Plane(w,h),Plane(w,h)};
    for(int c=0;c<3;++c){
        for(std::size_t i=0;i<emission.values.size();++i)glows[c][i]=(*source[c])[i]*smooth(.5f,.95f,image.gray[i]);
        glows[c]=blur(glows[c],side*.028f,cancel);
    }
    graphics::runParallelBatch(std::size_t(h),6,[&](std::size_t row){
        check(cancel);const float y=float(row);
        for(int x=0;x<w;++x){
            const auto i=row*w+x;
            const float far=depth.values.empty()?(1-smooth(.45f,1.05f,(y+.5f)/h))*(1-.8f*near[i]):1-depth[i];
            const float ramp=smooth(start,end,far);
            const float fog=1-std::exp(-3*haze*(.12f+.88f*ramp));
            result.fog[i]=fog;
            const float glow=std::exp(-std::hypot(x-lightX,y-lightY)/(side*.35f));
            float sum=0;
            if(length==0)sum=emission[i]*50.38f;
            else{
                const float stepX=(lightX-x)*length/96,stepY=(lightY-y)*length/96,jitter=hash(x*.618f+y*1.414f);
                for(int j=0;j<96;++j)sum+=emission.sample(x+stepX*(j+jitter),y+stepY*(j+jitter))*weights[j];
            }
            result.beams[i]=sum/24*(1-.92f*near[i]);
            for(int c=0;c<3;++c){
                const float color=air[c][i]*std::lerp(1.f,tint[c],.6f)*.9f+tint[c]*glow*.45f;
                result.fogColor[c][i]=color*fog+glows[c][i]*haze*1.6f*(.3f+.7f*ramp);
            }
        }
    });
    return result;
}
}

bool isLighting(const Request& request){return request.kind==Kind::RimLight||request.kind==Kind::VolumetricLight;}
bool lightingIsIdentity(const Request& request){
    if(request.kind==Kind::RimLight){
        const auto& settings=request.settings.rim;
        return std::all_of(settings.lights.begin(),settings.lights.begin()+std::clamp(settings.count,1,3),
                           [](const RimLamp& light){return control(light.strength)==0;});
    }
    return control(request.settings.volume.haze)==0&&control(request.settings.volume.intensity)==0;
}

Result applyLighting(const Request& request,Result result){
    if(lightingIsIdentity(request))return result;
    auto scene=getScene(request);
    const auto& cancel=request.limits.cancelled;
    if(request.kind==Kind::RimLight&&scene->subject.empty())throw std::runtime_error("No foreground subject was detected");
    const auto& source=*request.source;
    if(request.deepSource&&(!request.deepSource->valid()||request.deepSource->width!=source.width||request.deepSource->height!=source.height))
        throw std::runtime_error("HDR filter source grid changed");
    const double factor=request.preview?std::min(1.,1536./std::max(source.width,source.height)):1.;
    const int w=std::max(1,int(std::lround(source.width*factor))),h=std::max(1,int(std::lround(source.height*factor)));
    if(estimatedWorkingBytes(w,h,request.deepSource?20:10)>request.limits.maxWorkingBytes)throw std::runtime_error("Lighting exceeds the filter memory budget");
    const ImagePlanes image(*scene,cancel);
    std::optional<VolumeField> volume;
    std::optional<std::array<Plane,3>> rim;
    if(request.kind==Kind::RimLight)rim=rimField(image,request.settings.rim,cancel);
    else volume=volumeField(*scene,image,request);
    RGB beamTint{};
    for(int c=0;c<3;++c)beamTint[c]=2.5f*control(request.settings.volume.intensity)*float(srgbToLinear(unit(request.settings.volume.color[c])));
    auto light=[&](int x,int y){
        const auto sx=std::min(source.width-1,int((x+.5)*source.width/w));
        const auto sy=std::min(source.height-1,int((y+.5)*source.height/h));
        WidePixel base;
        if(request.deepSource)base=request.deepSource->pixel(sx,sy);
        else{const auto p=source.pixel(sx,sy);base={p.r/255.f,p.g/255.f,p.b/255.f,p.a/255.f};}
        const auto a=base[3];if(a<=0)return base;
        const float selection=request.selection?request.selection->coverage->pixel(sx-request.selection->originX,sy-request.selection->originY)/255.f:1;
        if(selection==0)return base;
        const float px=float((x+.5)*scene->sample->width/w-.5),py=float((y+.5)*scene->sample->height/h-.5);
        for(int c=0;c<3;++c){
            const auto linear=srgbToLinear(base[c]/a);
            double amount,veiled=linear;
            if(rim)amount=(*rim)[c].sample(px,py);
            else{
                veiled=linear*(1-volume->fog.sample(px,py))+volume->fogColor[c].sample(px,py);
                amount=volume->beams.sample(px,py)*beamTint[c];
            }
            const double lit=veiled+std::max(0.,1-veiled)*(1-std::exp(-std::max(0.,amount)));
            const auto converted=float(linearToSrgb(lit)*a);
            const float low=request.deepSource&&request.deepSource->isHdr()?-65504.f*a:0;
            const float high=request.deepSource&&request.deepSource->isHdr()?65504.f*a:a;
            base[c]=std::clamp(std::lerp(base[c],converted,selection),low,high);
        }
        return base;
    };
    if(request.deepSource){
        auto deep=makeDeep(w,h,request.deepSource->format);
        graphics::runParallelBatch(std::size_t(h),6,[&](std::size_t y){check(cancel);for(int x=0;x<w;++x)deep->setPixel(x,int(y),light(x,int(y)));});
        result.deep=deep;result.raster=rasterFromDeep(*deep,cancel);
    }else{
        std::vector<Pixel> pixels(std::size_t(w)*h);
        graphics::runParallelBatch(std::size_t(h),6,[&](std::size_t y){
            check(cancel);
            auto byte=[](float v){return uint8_t(std::lround(v*255));};
            for(int x=0;x<w;++x){const auto p=light(x,int(y));pixels[y*w+x]={byte(p[0]),byte(p[1]),byte(p[2]),byte(p[3])};}
        });
        result.raster=Raster::fromRgba(w,h,reinterpret_cast<const uint8_t*>(pixels.data()),std::size_t(w)*4,cancel);
    }
    result.previewScale=factor;result.changed=true;result.implementation="CPU lighting with DepthMini and BiRefNet";
    return result;
}

std::shared_ptr<const Raster> extractLightingDepth(const Request& request){
    auto scene=getScene(request);
    if(scene->depth.empty())throw std::runtime_error(scene->depthIssue.empty()?"A depth map could not be generated for this image":scene->depthIssue);
    const auto& cancel=request.limits.cancelled;
    const auto depth=composedDepth(*scene,ImagePlanes(*scene,cancel),request.settings.volume,cancel);
    const auto w=request.source->width,h=request.source->height;
    if(estimatedWorkingBytes(w,h,4)>request.limits.maxWorkingBytes)throw std::runtime_error("Depth extraction exceeds the filter memory budget");
    std::vector<Pixel> pixels(std::size_t(w)*h);
    graphics::runParallelBatch(std::size_t(h),6,[&](std::size_t y){
        check(cancel);
        for(int x=0;x<w;++x){
            const auto value=uint8_t(std::lround(255*unit(depth.sample(float((x+.5)*depth.width/w-.5),float((y+.5)*depth.height/h-.5)))));
            pixels[y*w+x]={value,value,value,255};
        }
    });
    return Raster::fromRgba(w,h,reinterpret_cast<const uint8_t*>(pixels.data()),std::size_t(w)*4,cancel);
}

}
