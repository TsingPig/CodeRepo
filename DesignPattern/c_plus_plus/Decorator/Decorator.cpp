#include <iostream>

// 抽象组件
class Player {
public:
    virtual void Equip() = 0;
    virtual ~Player() {}
};

// 具体组件
class BasicPlayer : public Player {
public:
    void Equip() override {
        std::cout << "Basic equipment equipped" << std::endl;
    }
};

// 抽象装饰类
class PlayerDecorator : public Player {
protected:
    Player* player;

public:
    PlayerDecorator(Player* p) : player(p) {}

    void Equip() override {
        if (player != nullptr) {
            player->Equip();
        }
    }

    virtual ~PlayerDecorator() {
        delete player;
    }
};

// 具体装饰类
class SwordDecorator : public PlayerDecorator {
public:
    SwordDecorator(Player* p) : PlayerDecorator(p) {}

    void Equip() override {
        PlayerDecorator::Equip();
        std::cout << "Sword equipped" << std::endl;
    }
};

class ShieldDecorator : public PlayerDecorator {
public:
    ShieldDecorator(Player* p) : PlayerDecorator(p) {}

    void Equip() override {
        PlayerDecorator::Equip();
        std::cout << "Shield equipped" << std::endl;
    }
};

// 使用
int main() {
    // 创建基本玩家
    Player* basicPlayer = new BasicPlayer();

    // 添加剑和盾的装备
    Player* decoratedPlayer = new ShieldDecorator(new SwordDecorator(basicPlayer));

    // 游戏中动态装备
    decoratedPlayer->Equip();

    // 释放资源
    delete decoratedPlayer;

    return 0;
}
