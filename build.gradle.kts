// Plugin versions live on the buildscript classpath (not in plugins{}) because
// this machine's Gradle cache holds the artifacts but not every plugin marker;
// offline builds must not need the markers. Same arrangement as the dreamlite
// app, which builds here.
buildscript {
    repositories {
        google()
        mavenCentral()
    }
    dependencies {
        classpath("com.android.tools.build:gradle:9.0.1")
        classpath("org.jetbrains.kotlin:kotlin-gradle-plugin:2.2.10")
        classpath("org.jetbrains.kotlin:compose-compiler-gradle-plugin:2.2.10")
    }
}
