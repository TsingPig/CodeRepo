#include <unordered_map>
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>

class LexicalAnalyzer
{
public:
    enum TokenType
    {
        Identifier = 0, // 标识符
        Constant = 24,  // 常量
        Other           // 其他
    };

public:
    // 标识符、常量及保留字识别函数
    const std::unordered_map<std::string, int> dic =
        {
            {"main",1,},
            {"int", 2},
            {"if", 3},
            {"else", 4},
            {"while", 5},
            {"do", 6},
            {"<", 7},
            {">", 8},
            {"!=", 9},
            {">=", 10},
            {"<=", 11},
            {"==", 12},
            {",", 13},
            {";", 14},
            {"(", 15},
            {")", 16},
            {"{", 17},
            {"}", 18},
            {"+", 19},
            {"-", 20},
            {"*", 21},
            {"/", 22},
            {"=", 23}};

public:
    bool isLittleLetter(const char ch) { return 'a' <= ch && ch <= 'z'; }
    bool isdigit(const char ch) {return '0' <= ch && ch <= '9' || ch == '.';}
    // 读入字符串进行分割, 提取出dic中的标识符、常量及保留字
    std::string analyze(const std::string &codeStr)
    {
        std::string curWord;
        std::string res;
        TokenType type = Other;

        // 执行操作函数，封装方便服用
        auto func = [&](int dicVal) -> void
        {
            std::string tmp = "(" + curWord + "," + std::to_string(dicVal) + ")\n";
            res.append(tmp);
            curWord.clear(); // 清空
            type = Other;
        };

        for (int i = 0; i < codeStr.length(); i++)
        {   
            const auto& ch = codeStr[i];
            if (ch == '\n' || ch == ' ' || ch == '\t')  
            {  
                // 分割出串，进行判断
                if (dic.find(curWord) != dic.end())
                    func(dic.at(curWord));
                else if(!curWord.empty())
                    // 没找到，不是保留字。需要判断是整数还是标识符
                    func(type);
            }
            else if (isLittleLetter(ch))
            {
                if (type == Constant)
                    // 抛出异常，标识符不可以以数字开头
                    throw std::runtime_error("Identifier can not start with number: " + curWord);
                else if (curWord.empty())
                    type = Identifier;
                curWord.push_back(ch);
            }
            else if (isdigit(ch))
            {
                if (curWord.empty())
                    type = Constant;
                curWord.push_back(ch);
            }
            else
            {
                
                // 遇到操作符，清空上一次的串
                if (!curWord.empty()){
                    if (dic.find(curWord) != dic.end())
                        func(dic.at(curWord));
                    else
                        func(type);
                }

                // 注解开始
                curWord.push_back(ch);
                
                if (i + 1 < codeStr.length())
                    curWord.push_back(codeStr[i + 1]);

                if (dic.find(curWord) != dic.end()){
                    func(dic.at(curWord));
                    i++;
                    continue;   
                }
                curWord.pop_back();
                if (dic.find(curWord) != dic.end()){
                    func(dic.at(curWord));
                    continue;
                }
                throw std::runtime_error("Unknown operator: " + curWord);
            }
        }
        return res;
    }
};


class IO{
public:
    static std::string readFromFile(const std::string& path){
        std::ifstream fin(path);
        if (!fin)
            throw std::runtime_error("Unable to open s.txt");
        std::stringstream buffer;
        buffer << fin.rdbuf();
        std::string res = buffer.str();
        fin.close();
        return res;
    }
    static void writeToFile(const std::string& path, const std::string& content){
        std::ofstream fo(path);
        if(!fo)
            throw std::runtime_error("Unable to open s.txt");
        fo << content;
        fo.close();
    }
};

int main()
{

    LexicalAnalyzer la;
    IO::writeToFile("F:/--CodeRepo/--CodeRepo/Project/c_plus_plus/Complie_Principle/result.txt", 
        la.analyze(IO::readFromFile("F:/--CodeRepo/--CodeRepo/Project/c_plus_plus/Complie_Principle/s.txt")));
    return 0;
}