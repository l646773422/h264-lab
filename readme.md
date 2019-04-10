# 这是一个视频编码练习仓库

基于大佬[lieff](<https://github.com/lieff>)的mini264来做各视频编码模块学习训练仓库，[原repo](<https://github.com/lieff/minih264>)。

构建指南

```powershell
mkdir build
cd build
cmake ..\src
```

将sequence中的**foreman.zip**解压后即可得到测试使用序列。

使用下面的命令即可

```powershell
encode_app.exe --output <bit-stream name> --input <sequence name>  
```
