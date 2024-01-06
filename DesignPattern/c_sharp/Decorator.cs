using System;

// 抽象组件
public abstract class Player
{
    public abstract void Equip();
}

// 具体组件
public class BasicPlayer : Player
{
    public override void Equip()
    {
        Console.WriteLine("Basic equipment equipped");
    }
}

// 抽象装饰类
public abstract class PlayerDecorator : Player
{
    protected Player player;

    public PlayerDecorator(Player player)
    {
        this.player = player;
    }

    public override void Equip()
    {
        if (player != null)
        {
            player.Equip();
        }
    }
}

// 具体装饰类
public class SwordDecorator : PlayerDecorator
{
    public SwordDecorator(Player player) : base(player) { }

    public override void Equip()
    {
        base.Equip();
        Console.WriteLine("Sword equipped");
    }
}

public class ShieldDecorator : PlayerDecorator
{
    public ShieldDecorator(Player player) : base(player) { }

    public override void Equip()
    {
        base.Equip();
        Console.WriteLine("Shield equipped");
    }
}

// 使用
public class Game
{
    static void Main(string[] args)
    {
        // 创建基本玩家
        Player basicPlayer = new BasicPlayer();

        // 添加剑和盾的装备
        Player decoratedPlayer = new SwordDecorator(new ShieldDecorator(basicPlayer));

        // 游戏中动态装备
        decoratedPlayer.Equip();

        Console.ReadKey(); // 等待用户按下键盘任意键，以便在终端看到输出结果
    }
}
