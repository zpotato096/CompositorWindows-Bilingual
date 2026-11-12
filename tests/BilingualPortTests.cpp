#include "effects/Adjustments.h"
#include "filters/PixelFilters.h"
#include "persistence/ProjectStore.h"
#include "ui/InterfaceLanguage.h"
#include "ui/ImportActions.h"
#include "ui/MainWindow.h"

#include <QAction>
#include <QApplication>
#include <QDialog>
#include <QDialogButtonBox>
#include <QDir>
#include <QFileInfo>
#include <QElapsedTimer>
#include <QLabel>
#include <QMessageBox>
#include <QPushButton>
#include <QSettings>
#include <QStatusBar>
#include <QTabWidget>
#include <QTemporaryDir>
#include <QTest>
#include <QTimer>
#include <array>
#include <cstdio>
#include <stdexcept>

namespace {
using namespace compositor;

void check(bool condition,const char* message){
    if(!condition)throw std::runtime_error(message);
}

void await(const std::function<bool()>& condition,const char* message,int timeout=8000){
    QElapsedTimer time;time.start();
    while(!condition()&&time.elapsed()<timeout)QTest::qWait(25);
    check(condition(),message);
}

std::shared_ptr<const Raster> image(){
    std::array<uint8_t,32*32*4> pixels{};
    for(int y=0;y<32;++y)for(int x=0;x<32;++x){
        const auto i=std::size_t(y*32+x)*4;
        const auto noise=((x+y)%2)*20;
        pixels[i]=uint8_t((x<16?55:165)+noise);
        pixels[i+1]=uint8_t(100+noise);
        pixels[i+2]=uint8_t(120-noise);
        pixels[i+3]=255;
    }
    return Raster::fromRgba(32,32,pixels.data(),32*4);
}

Document sample(){
    Document document;
    document.id=newId();document.width=32;document.height=32;
    Layer layer;
    layer.id=newId();layer.name="Source";layer.raster=image();layer.transform={0,0,32,32};
    document.layers.push_back(std::move(layer));
    return document;
}

void details(const QTemporaryDir& temporary){
    for(auto [name,version]:{std::pair{"Sharpen",19},std::pair{"Reduce Noise",21}}){
        auto document=sample();
        const auto source=document.layers.front().raster;
        const auto json=effects::defaultAdjustmentJson(name);
        const auto adjusted=effects::applyAdjustment(source,json);
        check(adjusted->rgba()!=source->rgba(),"Detail adjustment did not change any pixels");
        check(adjusted->pixel(15,15).a==source->pixel(15,15).a,"Detail adjustment changed alpha");
        Layer adjustment;
        adjustment.id=newId();adjustment.name=name;adjustment.transform={0,0,32,32};adjustment.adjustmentJson=json;
        document.layers.push_back(adjustment);
        const auto path=std::filesystem::path(temporary.filePath(QString::fromUtf8(name)+".comp").toStdWString());
        ProjectStore store(makeWicProjectCodec());store.save(path,document,adjustment.id);
        const auto restored=store.load(path);
        check(restored.readVersion==version,"Incorrect detail adjustment format version");
        check(restored.document.layers.back().adjustmentJson==json,"Detail settings did not survive reopening");
        const auto reapplied=effects::applyAdjustment(restored.document.layers.front().raster,restored.document.layers.back().adjustmentJson);
        check(reapplied->rgba()==adjusted->rgba(),"Reopened detail adjustment rendered differently");
    }
}

void lights(){
    const auto source=image();
    auto scene=std::make_shared<filters::LightingScene>();
    scene->sample=source;
    scene->subject.resize(32*32);scene->depth.resize(32*32);
    for(int y=0;y<32;++y)for(int x=0;x<32;++x){
        scene->subject[y*32+x]=(x-16)*(x-16)+(y-16)*(y-16)<100?1.f:0.f;
        scene->depth[y*32+x]=float(y)/31;
    }
    filters::Request request;request.source=source;request.transform={0,0,32,32};
    request.prepareLighting=[scene](const auto&){return scene;};
    auto coverage=std::make_shared<GrayRaster>(GrayRaster{32,32,std::vector<uint8_t>(32*32,255)});
    coverage->pixels[0]=0;
    request.selection=filters::SourceSelection{coverage,0,0};
    for(auto kind:{filters::Kind::RimLight,filters::Kind::VolumetricLight}){
        request.kind=kind;
        const auto result=filters::apply(request);
        check(result.changed&&result.raster->rgba()!=source->rgba(),"Lighting did not affect a synthetic subject");
        check(result.raster->pixel(0,0)==source->pixel(0,0),"Lighting changed pixels outside the selection");
        check(result.raster->pixel(20,20).a==255,"Lighting changed alpha");
    }
    const auto extracted=filters::extractLightingDepth(request);
    check(extracted->pixel(0,28).r>extracted->pixel(0,3).r,"Extracted depth lost near/far ordering");
    auto hdr=makeDeep(32,32);
    for(int y=0;y<32;++y)for(int x=0;x<32;++x)hdr->setPixel(x,y,{2,1.5f,1,1});
    request.kind=filters::Kind::RimLight;request.deepSource=hdr;
    const auto wide=filters::apply(request);
    check(wide.deep&&wide.deep->isHdr()&&wide.deep->pixel(20,20)[0]>1,"Lighting clipped HDR highlights");
}

void lightingUi(){
    MainWindow window(true);
    window.show();
    auto document=sample();
    std::array<Pixel,64*64> data{};
    for(int y=0;y<64;++y)for(int x=0;x<64;++x)
        if((x-32)*(x-32)+(y-32)*(y-32)<625)data[y*64+x]={uint8_t(x*3),uint8_t(y*3),120,255};
    document.width=document.height=64;
    document.layers[0].raster=Raster::fromRgba(64,64,reinterpret_cast<const uint8_t*>(data.data()),64*4);
    document.layers[0].transform={0,0,64,64};
    auto& project=window.addProject(std::move(document));
    auto run=[&](const char* id,const char* title){
        auto* action=window.findChild<QAction*>(id);
        check(action&&action->isEnabled(),"Lighting command not enabled");
        action->trigger();
        auto* panel=window.findChild<QDialog*>("filterPanel");
        check(panel&&panel->windowTitle()==title,"Lighting panel did not open");
        return panel;
    };
    auto* rim=run("command.filter.rim_light","Rim Light");
    auto* buttons=rim->findChild<QDialogButtonBox*>();
    check(buttons,"Missing Rim Light actions");
    auto* apply=buttons->button(QDialogButtonBox::Apply);
    await([&]{return apply&&apply->isEnabled();},"Rim Light preview did not complete");
    apply->click();
    await([&]{return project.history.canUndo();},"Rim Light did not create an undo step");
    QTest::qWait(150);
    auto* volume=run("command.filter.volumetric_light","Volumetric Light");
    auto* extract=volume->findChild<QPushButton*>("extractDepth");
    check(extract,"Missing Extract Depth button");extract->click();
    await([&]{return project.document&&project.document->layers.size()==2;},"Extract Depth did not insert a layer",15000);
    check(project.document->layers.back().name=="Depth"&&project.active==project.document->layers.back().id,"Depth layer was not selected");
    check(project.history.undoName()=="Extract Depth","Depth extraction did not create a named undo step");
}

void projectChanges(const QTemporaryDir& temporary){
    MainWindow window(true);window.show();
    auto& project=window.addProject(sample());
    const auto name=temporary.filePath("Watched.comp");
    const auto path=std::filesystem::path(name.toStdWString());
    window.setProperty("testNativeSavePath",name);
    auto* save=window.findChild<QAction*>("command.file.save");
    check(save&&save->isEnabled(),"Save Project command not available");save->trigger();
    await([&]{return project.path==name;},"Initial save did not finish");
    const auto active=project.active;
    ProjectStore store(makeWicProjectCodec());
    auto disk=*project.document;disk.layers.front().name="From another process";
    store.save(path,disk,active);
    await([&]{return project.document->layers.front().name=="From another process";},"External project replacement did not reload");
    check(project.active==active&&!project.history.modified(),"External reload lost active layer or left history dirty");
    auto imagePath=path/L"images"/std::filesystem::path(QString::fromStdString(active+".png").toStdWString());
    auto codec=makeWicProjectCodec();
    const auto replacement=Raster::filled(32,32,{12,34,56,255});
    codec.writeColor(imagePath,*replacement);
    await([&]{return project.document->layers.front().raster->pixel(0,0)==Pixel{12,34,56,255};},"Asset-only edit did not reload");
    project.history.begin("Rename",project.document,active);project.document->layers.front().name="My unsaved layer";
    project.history.end(project.document,active);
    bool prompt=false;
    QTimer choice;
    choice.setInterval(25);
    QObject::connect(&choice,&QTimer::timeout,&window,[&]{
        for(auto* alert:window.findChildren<QMessageBox*>())if(alert->isVisible()&&alert->windowTitle()=="Project changed on disk"){
            for(auto* button:alert->buttons())if(button->text()=="Keep My Changes"){prompt=true;button->click();return;}
        }
    });
    choice.start();disk.layers.front().name="Second disk revision";store.save(path,disk,active);
    await([&]{return prompt;},"Unsaved changes did not prompt before reload");choice.stop();
    check(project.document->layers.front().name=="My unsaved layer"&&project.history.modified(),"Keep My Changes discarded unsaved data");
}

void macProject(const QTemporaryDir& temporary){
    auto doc=sample();
    auto second=Raster::filled(32,32,{30,49,5,127});
    auto& layer=doc.layers.front();const auto animated=layer.id;
    layer.animation=GifAnimation{{{layer.raster,83},{second,179}},0};
    Layer lens;lens.id=newId();lens.name="Lens Effects";lens.transform={0,0,32,32};
    lens.adjustmentJson=effects::defaultAdjustmentJson("Lens Effects");
    doc.layers.push_back(lens);
    const auto path=std::filesystem::path(temporary.filePath("Mac copy.comp").toStdWString());
    ProjectStore store(makeWicProjectCodec());
    store.saveMacCompatible(path,doc,animated);
    const auto reopened=store.openObserved(path);
    check(reopened.project&&reopened.project->readVersion==21,"Mac copy did not use v21");
    const auto& layers=reopened.project->document.layers;
    check(layers[0].animation&&layers[0].animation->frames[1].delayMs==179,"Mac GIF timing was not restored");
    check(layers[0].animation->frames[1].raster->rgba()==second->rgba(),"Mac GIF did not preserve edited color and alpha");
    check(layers[1].adjustmentJson==lens.adjustmentJson,"Mac copy lost editable Lens Effects");
    check(!store.observe(path,reopened.fingerprint).project,"Unchanged package was reported as changed");
}

void webp(const QTemporaryDir& temporary){
    const auto name=temporary.filePath("透明图像.webp");
    QImage image(20,30,QImage::Format_ARGB32);
    image.fill(QColor(255,0,100,64));
    check(image.save(name,"WEBP",100),"WebP codec plugin is missing");
    const auto result=ui::decodeImportImage(std::filesystem::path(name.toStdWString()),{});
    check(result.image.width==20&&result.image.height==30&&result.image.pixels[3]==64&&result.metadata.decoder=="WebP",
        "WebP import did not preserve dimensions, alpha, and decoder metadata");
}

void layerPng(const QTemporaryDir& temporary){
    auto doc=sample();
    Layer top;top.id=newId();top.name="Trim me";top.transform={6,8,8,9};top.raster=Raster::filled(8,9,{230,10,20,255});
    doc.layers.push_back(top);
    MainWindow window(true);window.show();
    auto& project=window.addProject(std::move(doc));
    const auto path=temporary.filePath("trimmed.png");
    window.setProperty("testNativeSavePath",path);
    auto* action=window.findChild<QAction*>("command.layer.export_png");
    check(action&&action->isEnabled(),"Layer PNG export action was not registered");
    action->trigger();
    QImage saved(path);
    check(saved.size()==QSize(8,9)&&saved.pixelColor(2,3).red()==230,"Layer PNG was not isolated and trimmed");
    check(!project.history.modified(),"Layer PNG export changed the editable document");
}

void language(bool chinese,const QTemporaryDir& temporary){
    const QString code=chinese?"zh-CN":"en";
    QSettings().setValue("interface/language",code);
    ui::initializeInterfaceLanguage();
    MainWindow window(true);
    window.show();QTest::qWait(150);
    auto* heading=window.findChild<QLabel*>("newCanvasHeading");
    check(heading&&heading->text()==(chinese?"新建画布":"New canvas"),"Welcome screen used the wrong language");
    auto* selected=window.findChild<QAction*>("interfaceLanguage."+code);
    check(selected&&selected->isChecked(),"Saved language not selected");
    auto* other=window.findChild<QAction*>(chinese?"interfaceLanguage.en":"interfaceLanguage.zh-CN");
    check(other,"Other language option missing");
    QTimer::singleShot(0,&window,[]{
        for(auto* widget:QApplication::topLevelWidgets()){
            if(auto* message=qobject_cast<QMessageBox*>(widget);message&&message->isVisible())
                message->accept();
        }
    });
    other->trigger();
    check(QSettings().value("interface/language").toString()==(chinese?"en":"zh-CN"),"Could not persist the selected language");
    const QString imagePath=temporary.filePath("Photo example.png");
    QImage input(32,32,QImage::Format_ARGB32);input.fill(Qt::yellow);
    check(input.save(imagePath),"Could not prepare the input PNG");
    window.openPath(imagePath);
    auto* tabs=window.findChild<QTabWidget*>();
    check(tabs,"Document tabs missing");
    for(int i=0;i<100&&!tabs->tabText(tabs->currentIndex()).startsWith("Photo example");++i)QTest::qWait(20);
    check(tabs->tabText(tabs->currentIndex()).startsWith("Photo example"),"Imported image did not name the canvas");
    auto* filter=window.findChild<QAction*>("command.filter.sharpen");
    check(filter&&filter->text()==(chinese?"锐化…":"Sharpen…")&&filter->isEnabled(),"Sharpen command missing or not localized");
    filter->trigger();QTest::qWait(250);
    auto* panel=window.findChild<QDialog*>("filterPanel");
    check(panel&&panel->isVisible()&&panel->windowTitle()==(chinese?"锐化":"Sharpen"),"Localized Sharpen panel did not open");
    const auto capture=qEnvironmentVariable("COMPOSITOR_TEST_CAPTURE");
    if(!capture.isEmpty()){
        check(QDir().mkpath(capture),"Could not prepare the interface capture folder");
        const QDir folder(capture);
        check(window.grab().save(folder.filePath(code+"-editor.png")),"Could not capture the editor");
        check(panel->grab().save(folder.filePath(code+"-sharpen.png")),"Could not capture the Sharpen panel");
    }
    if(chinese){
        const auto boxes=panel->findChildren<QDialogButtonBox*>();
        check(!boxes.isEmpty()&&boxes.front()->button(QDialogButtonBox::Cancel)->text().contains("取消"),"Standard dialog button not localized");
    }
    panel->reject();
}
}

int main(int argc,char** argv){
    QApplication application(argc,argv);
    application.setOrganizationName("CompositorLocalTests");application.setApplicationName("BilingualPort");
    QTemporaryDir temporary;
    QSettings::setDefaultFormat(QSettings::IniFormat);
    QSettings::setPath(QSettings::IniFormat,QSettings::UserScope,temporary.path());
    QSettings::setPath(QSettings::IniFormat,QSettings::SystemScope,temporary.path());
    try{
        check(temporary.isValid()&&argc==2,"Test mode or temporary folder missing");
        const auto mode=QString::fromUtf8(argv[1]);
        if(mode=="detail_layers")details(temporary);
        else if(mode=="lighting_kernels")lights();
        else if(mode=="lighting_ui")lightingUi();
        else if(mode=="external_changes")projectChanges(temporary);
        else if(mode=="mac_project")macProject(temporary);
        else if(mode=="webp")webp(temporary);
        else if(mode=="layer_png")layerPng(temporary);
        else language(mode=="chinese",temporary);
    }catch(const std::exception& error){
        std::fprintf(stderr,"%s\n",error.what());
        return 1;
    }
}
