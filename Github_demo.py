'''
您遇到的错误表明，您的项目中包含一个大于 GitHub 文件大小限制（100MB）的文件，即 huri/components/exe/best.pt，这个文件大小为 333.83 MB。GitHub 默认不允许上传超过 100MB 的文件，但您可以使用 Git LFS（Git Large File Storage）来管理大文件。

解决方案：使用 Git LFS 处理大文件
Git LFS 是一个 Git 扩展，用于存储大型文件和二进制文件。以下是设置和使用 Git LFS 的步骤：

1. 安装 Git LFS
如果您尚未安装 Git LFS，请在终端中运行以下命令来安装它：

Windows：下载并安装 Git LFS，链接：Git LFS 下载
MacOS：使用 Homebrew 安装
bash
复制代码
brew install git-lfs
Linux：可以使用包管理器安装，例如
bash
复制代码
sudo apt-get install git-lfs  # Debian/Ubuntu 系统
2. 初始化 Git LFS
在安装 Git LFS 后，初始化它：

bash
复制代码
git lfs install
3. 跟踪大文件
使用 Git LFS 来跟踪需要上传的大文件。运行以下命令以告诉 Git LFS 跟踪 best.pt 文件：

bash
复制代码
git lfs track "huri/components/exe/best.pt"
此命令会在您的项目中创建一个 .gitattributes 文件，该文件包含 Git LFS 跟踪的文件信息。

4. 提交 .gitattributes 文件
添加并提交 .gitattributes 文件，以便 Git 知道哪些文件使用 Git LFS：

bash
复制代码
git add .gitattributes
git commit -m "Track large files using Git LFS"
5. 重新添加和提交大文件
重新添加大文件并提交更改：

bash
复制代码
git add huri/components/exe/best.pt
git commit -m "Add large file with Git LFS"
6. 推送到 GitHub
现在可以将所有更改推送到 GitHub。Git LFS 会处理大文件的推送：

bash
复制代码
git push origin main
注意事项
Git LFS 的存储限制：GitHub 提供了免费计划的 Git LFS 存储，但可能有限制。如果项目中有多个大文件或超大的文件，可能需要购买更多的存储空间。

其他成员的 LFS 设置：如果其他人也需要克隆此项目并使用这些大文件，他们也需要安装 Git LFS。

完成这些步骤后，您的大文件应成功推送到 GitHub。








'''