#include <iostream>
#include <memory>

// 抽象组件
class Player
{
public:
    virtual void Equip() = 0;
    virtual ~Player() {}
};

// 具体组件
class BasicPlayer : public Player
{
public:
    void Equip() override
    {
        std::cout << "Basic equipment equipped" << std::endl;
    }
};

// 抽象装饰类
class PlayerDecorator : public Player
{
protected:
    std::unique_ptr<Player> player;

public:
    PlayerDecorator(std::unique_ptr<Player> p) : player(std::move(p)) {}

    void Equip() override
    {
        if (player != nullptr)
        {
            player->Equip();
        }
    }
};

// 具体装饰类
class SwordDecorator : public PlayerDecorator
{
public:
    SwordDecorator(std::unique_ptr<Player> p) : PlayerDecorator(std::move(p)) {}

    void Equip() override
    {
        PlayerDecorator::Equip();
        std::cout << "Sword equipped" << std::endl;
    }
};

class ShieldDecorator : public PlayerDecorator
{
public:
    ShieldDecorator(std::unique_ptr<Player> p) : PlayerDecorator(std::move(p)) {}

    void Equip() override
    {
        PlayerDecorator::Equip();
        std::cout << "Shield equipped" << std::endl;
    }
};


class HelmetDecorator : public PlayerDecorator
{
public:
    HelmetDecorator(std::unique_ptr<Player> p) : PlayerDecorator(std::move(p)) {}

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
    std::unique_ptr<Player> basicPlayer = std::make_unique<BasicPlayer>();

    // 添加剑和盾的装备
    std::unique_ptr<Player> decoratedPlayer = std::make_unique<ShieldDecorator>(
        std::make_unique<SwordDecorator>(std::move(basicPlayer)));

    /*
    std::move(basicPlayer) 是使用 C++ 中的 std::move 函数将 basicPlayer
    的所有权（ownership）转移到另一个地方。这通常用于移动语义，
    即在转移资源的同时保持程序的性能。在这里，它的目的是将 basicPlayer 的
    所有权传递给 std::unique_ptr 的构造函数。

    在 C++ 中，当你使用 std::unique_ptr 的构造函数时，
    通常需要传递一个指针。然而，你不希望简单地将指针直接传递给构造函数，
    因为这样将会导致两个 std::unique_ptr 共享同一个资源，这是不允许的。

    通过使用 std::move(basicPlayer)，你告诉编译器将 basicPlayer 的所有权
    （右值引用）传递给 std::unique_ptr 的构造函数。
    这样一来，std::unique_ptr 就拥有了对 basicPlayer 所指向对象的唯一所有权。
    */

    // 游戏中动态装备
    decoratedPlayer->Equip();

    std::cout << "Game End!" << std::endl;

    // 添加头盔
    decoratedPlayer = std::make_unique<HelmetDecorator>(std::move(decoratedPlayer));
    decoratedPlayer->Equip();
    // 不需要手动释放内存，智能指针会自动处理
    return 0;
}
