#　使用GitHub开发

本文件主要内容为使用git仓库的开发步骤。

如果你未曾使用过GitHub，请参考GitHub上编写得当的[快速指南](https://docs.github.com/en/github/getting-started-with-github/quickstart)，其解释了基础内容。

## 初始设置

如果你以前没有使用过Github，你需要至少进行一次这些设置。请浏览快速上手指南[设置Git](https://docs.github.com/en/github/getting-started-with-github/set-up-git)页面来启用你的git。接下来你需要fork一个仓库。在那之后，请参考后文Life of a Pull Request来了解常见的工作流程。

### 配置SSH访问

最容易配置的Github仓库访问方式是使用SSH密匙。为此，你需要一个SSH密匙对，最好是高强度的密匙对。你如果有意愿的话，可以为不同的网站使用不同的密匙。在这个例子里，我们只会创造一个用于GitHub的密匙。

通过执行以下命令创造`~/.ssh/id_rsa_github`文件。（此处以及以下其他地方，{{X}}是给你的邮件地址、用户名保留的占位符）

```bash
ssh-keygen -t rsa -b 4096 -C "{{EMAIL}}" -f ~/.ssh/id_rsa_github
```

前往你的 [SSH and GPG keys](https://github.com/settings/keys)设置页面，并粘贴你*公匙*（以`.pub`结尾的）的内容。该内容可通过以下命令输出：

```bash
cat ~/.ssh/id_rsa_github.pub
```

为在SSH到github.com时使用指定的密匙，你可以使用以下命令添加如下配置片段到你的.ssh/config文件。

```bash
cat >> ~/.ssh/config <<EOF

Host github.com
  Hostname github.com
  IdentityFile ~/.ssh/id_rsa_github
  IdentitiesOnly yes
EOF
```

`IdentitiesOnly yes`确保仅在与GitHub连接时使用提供的IdentityFile。

### Fork你的个人拷贝

JPEG XL的代码位于[此仓库](https://github.com/libjxl/libjxl)内。

开发者在GitHub上的工作流一般为：对某一仓库创建你自己的fork并将你自己的改动上传到那里。你可以从你自己的拷贝直接*向*上游仓库请求合并，无需在上游仓库创建分支。

在Github上[Fork代码仓库](https://docs.github.com/en/github/getting-started-with-github/fork-a-repo)来创建你对该代码仓库的拷贝。在这之后，你可以通过Pull Request提议在主仓库中收录你的改变。
完成之后，你可以在https://github.com/{{USERNAME}}/libjxl 访问你的仓库，

这里的{{USERNAME}}代表你的Github用户名。

### 通过Github获取JPEG XL的代码

你需要通过“克隆”的方式来将源代码下载到你的电脑上。这一过程涉及到了两个代码仓库，上游仓库（`libjxl/lbjxl`）以及你的fork分支（`{{USERNAME}}/libjxl`）。你通常应该不断从上有仓库获取新的改动，并把他们push到你的fork分支上去。将你的改动从你的fork分支添加到上有仓库需要通过网页界面的Pull Requests。

[Fork a repo](https://docs.github.com/en/github/getting-started-with-github/fork-a-repo)涵盖了很多细节，但在git remote里使用`upstream`指代共同的上有仓库，使用`origin`指代你自己的代码。本指南使用一种不同的命名策略，这一点在下方的例子里有所体现。

在本指南中，`origin`指的时共同的上游仓库，而用`myfork`指代你自己的fork分支。如有意愿，你可以给你的fork分支使用其他的名字。首先使用如下命令进行设置，请用你自己的Github用户名代替`{{USERNAME}}`。

```bash
git clone git https://github.com/libjxl/libjxl --recursive
cd libjxl
git remote set-url --push origin git@github.com:{{USERNAME}}/libjxl.git
git remote add myfork git@github.com:{{USERNAME}}/libjxl.git
git remote -vv
```

这些命令的作用如下：

 * 创建了一个代码仓库，并将`origin`设定为上游远程仓库
 * 将“push”的URL指向你自己的fork分支，并
 * 创建一个指向你自己fork拷贝的新的远程仓库

最后一步不是必须的。因为 `origin`的“fetch”URL指向共同的上游仓库，而“push”URL指向你自己的fork拷贝，因此从`origin`获取（fetch）代码，无关你fork拷贝仓库的内容，会一直从上游仓库获取到最新的改动。

准备第二个origin`myfork`的唯一用处是让你可以从位于另一台电脑上的fork拷贝仓库中下载未决的改动。比如，如果你使用多台电脑工作，每台电脑都使用了这套设置，你可以从一台电脑向你的fork拷贝push改动，之后在另一台电脑上从`myfork`fetch这些内容。

在那之后，请参考《Life of a Pull Request》了解常见的工作流程。

通用的[GitHub工作流指南](https://docs.github.com/en/github/getting-started-with-github/github-flow)适用于为本项目发送Pull Request。

这里所有的命令都假设你已根据上述内容，使用git获取了代码。

### 同步到最新版本

```bash
git fetch origin
```

最新的上游版本现已处在`origin/main`上，而你的本地分支不会被此命令修改。

### 建立一个新的分支

在开始一个新的改动之前，你需要一个本地分支。每一个分支都代表一个包含了各个commit的列表，这一列表之后可以被合并为一个单独的合并请求。总的来说，一个分支代表一次代码审查，但每个分支里面可以有多个不同的commit。

```bash
git checkout origin/main -b mybranch
```

这一命令会新建一个跟随`origin/main`的新分支`mybranch`。分支可以跟随任何远程或本地分支，一些工具会利用这一点。运行`gitbranch -vv`命令，会显示你拥有的所有分支、各分支跟随目标以及和目标相比，该分支领先或落后多少个commit。如果你在建立分支时没有设置跟随目标，你可以通过运行如下命令来加入或改变当前分支跟随的分支：`git branch --set-upstream-to=...`。

### 向你的分支添加改动

请参考网上大量的教程，比如：来自https://git-scm.com/doc网站的[基础](https://git-scm.com/book/en/v2/Git-Basics-Getting-a-Git-Repository)章节就是一个很好的起步指南。创建、修改或删除文件，并在git commit时附上信息。

commit信息是必要的。一个commit信息应该遵守50/72规则：

*   第一行不多于50个字母。
*   然后留白一行
*  剩下的文字应该在达到72个字母时换行

第一行应该明确说明你的commit，因为这是大多数工具会显示给用户的部分。诸如“一些修正”之类的第一行毫无帮助。请说明commit包含的内容以及改动原因。

我们遵守[Google C++ CodingStyle](https://google.github.io/styleguide/cppguide.html)。我们准备了一个[clang-format](https://clang.llvm.org/docs/ClangFormat.html) 配置文件，以供自动对代码进行排版，你可以通过`./ci.sh lint`帮助工具来调用它。

阅读[CONTRIBUTING.md](../CONTRIBUTING.md)来获取为libjxl做贡献的更多信息。

### 上传你的改动以供审查

第一步是对你要发送的内容进行本地审查。`gitg`是一个好用的Gtk UI工具，可用于审查本地改动，也可使用`tig`获取一个ncurses命令行交互界面。此外，你也可以在终端运行：

```bash
git branch -vv
```

来展示你本地分支的当前状态。具体来讲，因为你的本地分支在跟随origin/main（如输出所示），git会指出你正以一个commit领先于跟随目标。

```
* mybranch       e74ae1a [origin/main: ahead 1] Improved decoding speed by 40%
```

我们推荐你在上传之前，再一次和上游同步（`git fetchorigin`）然后运行`git branch -vv`来查看上游是否有新的改动。如果上游确有新改动，你会在输出中看见一个“behind”标志。

```
* mybranch       e74ae1a [origin/main: ahead 1, behind 2] Improved decoding speed by 40%
```

若想基于上游最新改动同步你的改动，你需要进行一次rebase：

```bash
git rebase
```

这一命令默认会将你当前分支的改动rebase到跟随分支之上。这种情况下，该命令会尝试将当前commit应用于最新的origin/main（该分支比我们自己的多两个commit），你的分支现在也会包含其内容。注意，你可能需要处理一些冲突。一个同时进行fetch和rebase的快捷方式是运行`git pull-r`，这里`-r`代表“rebase”并会将本地commit在远程commit之上进行rebase。

在上传补丁之前，确保你的补丁符合[贡献指南](../CONTRIBUTING.md) 并能[构建且通过测试](building_and_testing_zho-CN.md).

当你准备好发送你的分支以供审查时，将其上传到*你的*fork分支：

```bash
git push origin mybranch
```

这回将你的本地分支"mybranch" push到你的fork中的一个名为"mybranch"的远程分支。名字可以改动，但记住这是公开的。这会展示一个用于创建merge请求的url。

```
Enumerating objects:627, done.
Counting objects:100% (627/627), done.
Delta compression using up to 56 threads
Compressing objects:100% (388/388), done.
Writing objects:100% (389/389), 10.71 MiB | 8.34 MiB/s, done.
Total 389 (delta 236), reused 0 (delta 0)
emote:
remote:Create a pull request for 'mybranch' on GitHub by visiting:
remote:      https://github.com/{{USERNAME}}/libjxl/pull/new/mybranch
remote:
To github.com:{{USERNAME}}/libjxl.git
 * [new branch]      mybranch -> mybranch
```

### 更新子模块

本仓库使用子模块来引入第三方的外部库依赖。每一个子模块都以commit哈希值为根据，指向一个外部仓库的某一特定外部commit。就像常规的源代码文件一样，哈希值是当前分支和你获取的jpeg xl commit的一部分。

当改变分支或者进行`git rebase`时，git很不幸的*不会*自动将哈希值更新成目标分支或jpeg xl commit中的哈希值，也不会把第三方子模块的源文件更新到最新。这就是说，虽然git会更新你硬盘上的jpeg xl源代码文件到最新状态，但它会原样保留子模块哈希值以及工作区里的第三方文件，和你改变分支之前一致。这会在git diff里面体现出来，因为这和你切换到的分支相比，会被视为一个改动。git diff会显示哈希值的不同（就好像你改到老版本一样），它不会显示third_party目录中文件的内容的改变。

这一不匹配会至少带来以下两个问题：

*) jpeg xl代码库可能会因为第三方库的版本不匹配而编译失败比如，API发生了变化，或者添加、移除了一个子模块。

*) 在使用`commit -a`时，你或引入技术性改变、或引入与子模块无关的改变的commit会无意间改变子模块哈希值。除非你真的想改变第三方库的版本，这一结果是需要避免的。

为了解决这个问题，你需要在上述动作之后使用如下命令手动升级子模块（至少当子模块改变的时候）：

```
git submodule update --init --recursive
```

这里，init参数确保其会添加必要的新模块，而recursive参数则用于获取被现有子模块依赖的其他子模块。

如果你checkout到了一个不同的分支，你会在如下的信息中看到子模块的改变。

```
M       third_party/brotli
M       third_party/lcms
```

如果你此时进行rebase的话，你可能会误入一个更难解决的困境，这时`git submodule update --init --recursive`就会失败并报出如下错误：

```
Unable to checkout '35ef5c554d888bef217d449346067de05e269b30' in submodule path 'third_party/brotli'
```

这时，你需要使用force参数：

```
git submodule update --init --recursive --force
```

### 对你的merge请求进行迭代

若想要应用审查者的改动，你首先需要在你的分支中修正本地改动。你可以通过`git commit --amend file1 file2 file3 ...`或者`git commit --amend -a`来对你的commit进行改动，修正所有暂存文件中的改动。

在你准备好了新版本的"mybranch"分支以供再上传之后，你需要force push其到你fork拷贝的同一分支。因为你正在push同一个commit的不同版本（而不是在已有commit上提交另一个commit），你需要使用force执行操作来替换掉老版本。

```bash
git push origin mybranch --force
```

该merge请求现应已更新到最新的改动。

### merge你的改动

我们的merge策略是“rebase”，这代表我们不接受“merge” commit （有超过一个父级仓库的commit），只维护一个记录改动的线性历史。

在你最后一次rebase你的改动之后，很有可能主分支上被添加了其他改动。这些改动会和你的Pull Request冲突，所以你需要`git fetch`，`git rebase`并再一次push你的改动，这也需要你再进行一次连续的整合工作流来确认在添加最新改动之后，所有的测试仍会通过。

### 在本地测试待确认的Pull Request

如果你想在你的电脑上审查一个其他用户待确认的pull request，你可以使用如下命令fetch该带有merge请求的commit，注意，使用pull request的编号代替`NNNN`

```bash
git fetch origin refs/pull/NNNN/head
git checkout FETCH_HEAD
```

第一个命令会将该未决pull request的远程commit添加到你的本地git仓库中，并储存一个临时的，名为`FETCH_HEAD`出处引用标记。第二个命令会checkout该出处引用标记。在这之后，你可以在你的电脑上审查文件，给这一FETCH_HEAD创建本地分支，或者在这之上添加改动。
