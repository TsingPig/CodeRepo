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

/// <summary>
/// 抽象装饰类（Decorator）
/// </summary>
public abstract class Equipment : Player
{
    protected Player player;

    public Equipment(Player player)
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

/// <summary>
/// 具体装饰类（ConcreteDecorator）
/// </summary>
public class Sword : Equipment
{
    public Sword(Player player) : base(player) { }

    public override void Equip()
    {
        base.Equip();
        Console.WriteLine("Sword equipped");
    }
}

/// <summary>
/// 具体装饰类（ConcreteDecorator）
/// </summary>
public class Shield : Equipment
{
    public Shield(Player player) : base(player) { }

    public override void Equip()
    {
        base.Equip();
        Console.WriteLine("Shield equipped");
    }
}


public class Helmet : Equipment
{
    public Helmet(Player player) : base(player) { }

    public override void Equip()
    {
        base.Equip();
        Console.WriteLine("Helmet equipped");
    }
}
// 使用
public class Game
{
    static void Main(string[] args)
    {
        // 创建基本玩家
        Player basicPlayer = new BasicPlayer();

        // 动态添加剑和盾的装备
        Player equippedPlayer = new Sword(new Shield(basicPlayer));

        // 展示装备
        equippedPlayer.Equip();

        Console.WriteLine("GameEnd!");

        // 动态装备头盔
        new Helmet(equippedPlayer).Equip();

        /*
            Basic equipment equipped
            Shield equipped
            Sword equipped
            GameEnd!
            Basic equipment equipped
            Shield equipped
            Sword equipped
            Helmet equipped
        */
    }
}
