#include "InterfaceLanguage.h"

#include <QActionGroup>
#include <QByteArrayView>
#include <QCoreApplication>
#include <QFile>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLocale>
#include <QMenu>
#include <QMenuBar>
#include <QMessageBox>
#include <QSettings>
#include <QTranslator>
#include <utility>

static void initializeLanguageResources() { Q_INIT_RESOURCE(interface_language); }

namespace compositor::ui {
namespace {
constexpr auto languageKey = "interface/language";
constexpr auto titleKey = "compositorSourceTitle";
QJsonObject chinese;
QString activeLanguage = "en";

QString selectedLanguage(const QSettings& settings) {
    const auto choice = settings.value(languageKey).toString();
    if (choice == "en" || choice == "zh-CN") return choice;
    const auto system = QLocale::system();
    return system.language() == QLocale::Chinese && system.script() == QLocale::SimplifiedHanScript
        ? QStringLiteral("zh-CN") : QStringLiteral("en");
}

class WidgetTranslator final : public QTranslator {
public:
    using QTranslator::QTranslator;
    bool isEmpty() const override { return chinese.isEmpty(); }
    QString translate(const char* context, const char* source, const char*, int) const override {
        const auto name = QByteArrayView(context);
        if (name != "QDialogButtonBox" && name != "QPlatformTheme" && name != "QMessageBox"
            && name != "QColorDialog" && name != "QWidgetTextControl" && name != "QLineEdit") return {};
        return chinese.value(QString::fromUtf8(source)).toString();
    }
};

template<class Parent>
QMenu* createMenu(Parent* parent, const QString& source) {
    auto* menu = parent->addMenu(localized(source));
    menu->setProperty(titleKey, source);
    return menu;
}
}

void initializeInterfaceLanguage() {
    if (selectedLanguage(QSettings()) != "zh-CN") return;
    initializeLanguageResources();
    QFile file(":/translations/zh-CN.json");
    if (!file.open(QIODevice::ReadOnly)) return;
    chinese = QJsonDocument::fromJson(file.readAll()).object();
    if (chinese.isEmpty()) return;
    activeLanguage = "zh-CN";
    auto* app = QCoreApplication::instance();
    app->installTranslator(new WidgetTranslator(app));
}

QString localized(const QString& source) {
    const auto translation = chinese.value(source);
    if (translation.isString()) return translation.toString();
    if (chinese.isEmpty()) return source;
    if (source.endsWith(QChar(0x2026))) return localized(source.first(source.size() - 1)) + QChar(0x2026);
    if (source.contains('&')) {
        auto plain = source;
        plain.remove('&');
        const auto value = chinese.value(plain);
        if (value.isString()) return value.toString();
    }
    return source;
}

QStringList localizedList(const QStringList& source) {
    QStringList result;
    result.reserve(source.size());
    for (const auto& value : source) result.append(localized(value));
    return result;
}

QMenu* localizedMenu(QMenuBar* parent, const QString& source) { return createMenu(parent, source); }
QMenu* localizedMenu(QMenu* parent, const QString& source) { return createMenu(parent, source); }

QString menuSourceTitle(const QMenu* menu) {
    const auto source = menu->property(titleKey).toString();
    return source.isEmpty() ? menu->title() : source;
}

void addLanguageMenu(QMenu* help, QWidget* owner) {
    auto* menu = localizedMenu(help, "Language");
    menu->setObjectName("interfaceLanguage");
    auto* options = new QActionGroup(menu);
    const auto selected = selectedLanguage(QSettings());
    for (const auto& option : {std::pair{"English", "en"}, std::pair{"简体中文", "zh-CN"}}) {
        auto* action = menu->addAction(QString::fromUtf8(option.first));
        action->setObjectName(QStringLiteral("interfaceLanguage.") + QString::fromLatin1(option.second));
        action->setCheckable(true);
        options->addAction(action);
        action->setChecked(selected == QString::fromLatin1(option.second));
        QObject::connect(action, &QAction::triggered, owner, [owner, options, choice = QString::fromLatin1(option.second)] {
            QSettings settings;
            settings.setValue(languageKey, choice);
            settings.sync();
            if (settings.status() != QSettings::NoError) {
                for (auto* item : options->actions())
                    item->setChecked(item->objectName().endsWith(activeLanguage));
                QMessageBox::warning(owner, localized("Language"), localized("Could not save the language preference."));
                return;
            }
            if (choice != activeLanguage)
                QMessageBox::information(owner, localized("Language"), localized("Restart Compositor to apply the selected language."));
        });
    }
}

}
