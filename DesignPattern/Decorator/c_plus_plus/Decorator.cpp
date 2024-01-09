#include <iostream>

// 抽象组件
/*
纯虚函数在基类中没有具体的实现，而是由派生类来实现。
任何包含纯虚函数的类都被认为是抽象类，因此不能直接实例化。

C++中的virturl = 0 与C#中的abstract类似。
如果C++中virturl 有定义，与C#中的virtual类似。

*/
class Player
{
public:
    virtual void Equip() = 0;
    virtual ~Player() {}

    /*
    以便在通过基类指针删除派生类对象时，确保调用派生类的析构函数。这有助于防止内存泄漏。
    如果你不添加 virtual 关键字，那么基类的析构函数就不是虚函数。
    这样，当通过基类指针删除派生类对象时，只会调用基类的析构函数。
    在这种情况下，如果 PlayerDecorator 对象通过基类指针删除，它将只调用基类的析构函数，
    而不会调用派生类的析构函数。这可能导致派生类的资源泄漏。
    */
};

// 具体组件
class BasicPlayer : public Player
{
public:
    /*
    于明确指示派生类的成员函数要覆盖基类的虚函数。
    使用 override 关键字可以帮助在编译时检测错误，确保派生类中的函数确实是对基类虚函数的覆盖。
    */
    void Equip() override
    {
        std::cout << "Basic equipment equipped" << std::endl;
    }
};

// 抽象装饰类
class PlayerDecorator : public Player
{
protected:
    Player *player;

public:
    PlayerDecorator(Player *p) : player(p) {}

    void Equip() override
    {
        if (player != nullptr)
        {
            player->Equip();
        }
    }

    virtual ~PlayerDecorator()
    {
        delete player;
    }
};

// 具体装饰类
class SwordDecorator : public PlayerDecorator
{
public:
    SwordDecorator(Player *p) : PlayerDecorator(p) {}

    void Equip() override
    {
        PlayerDecorator::Equip();
        std::cout << "Sword equipped" << std::endl;
    }
};

class ShieldDecorator : public PlayerDecorator
{
public:
    ShieldDecorator(Player *p) : PlayerDecorator(p) {}

    void Equip() override
    {
        PlayerDecorator::Equip();
        std::cout << "Shield equipped" << std::endl;
    }
};

class HelmetDecorator : public PlayerDecorator
{
public:
    HelmetDecorator(Player *p) : PlayerDecorator(p) {}

    void Equip() override
    {
        PlayerDecorator::Equip();
        std::cout << "Helmet equipped" << std::endl;
    }
};

// 使用
int main()
{
    // 创建基本玩家
    Player *basicPlayer = new BasicPlayer();

    // 添加剑和盾的装备
    Player *decoratedPlayer = new ShieldDecorator(new SwordDecorator(basicPlayer));

    // 游戏中动态装备
    decoratedPlayer->Equip();

    // (new HelmetDecorator(decoratedPlayer))->Equip(); 会导致内存泄露

    // 另一种写法
    // auto *decoratedPlayer2 = new HelmetDecorator(decoratedPlayer);
    // decoratedPlayer2->Equip();

    std::cout << "Game End!" << std::endl;

    decoratedPlayer = new HelmetDecorator(decoratedPlayer);
    decoratedPlayer->Equip();
    // 释放资源
    delete decoratedPlayer;
    delete basicPlayer;
    // delete decoratedPlayer2;

    return 0;
}
