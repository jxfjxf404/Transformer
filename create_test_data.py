"""
============================================================
创建测试数据集
============================================================
生成小规模的英德翻译测试数据集，用于快速验证代码。

用法:
  python create_test_data.py

生成的数据保存在 data/test_translation/ 目录
============================================================
"""

import os
import random


def create_test_data(data_dir: str = "data/test_translation"):
    """
    创建测试数据集。

    参数:
        data_dir: 数据保存目录
    """
    print("=" * 60)
    print("创建测试数据集")
    print("=" * 60)

    os.makedirs(data_dir, exist_ok=True)

    train_en = [
        "a man is riding a horse",
        "a woman is playing the violin",
        "a dog is running in the park",
        "a cat is sleeping on the sofa",
        "a bird is flying in the sky",
        "a boy is reading a book",
        "a girl is drawing a picture",
        "a man is cooking in the kitchen",
        "a woman is dancing in the room",
        "a child is playing with toys",
        "two people are walking together",
        "three dogs are playing outside",
        "four birds are sitting on a tree",
        "five children are running in the field",
        "six cats are sleeping on the bed",
        "seven people are standing in line",
        "eight flowers are blooming in the garden",
        "nine cars are driving on the road",
        "ten students are studying in the library",
        "many birds are flying in the sky",
        "the sun is shining brightly",
        "the moon is full and bright",
        "the stars are twinkling at night",
        "the rain is falling softly",
        "the snow is falling gently",
        "the wind is blowing strongly",
        "the water is flowing smoothly",
        "the fire is burning hot",
        "the tree is growing tall",
        "the flower is blooming beautifully",
        "the bird is singing loudly",
    ]

    train_de = [
        "ein mann reitet ein pferd",
        "eine frau spielt geige",
        "ein hund rennt im park",
        "eine katze schläft auf dem sofa",
        "ein vogel fliegt am himmel",
        "ein junge liest ein buch",
        "ein mädchen zeichnet ein bild",
        "ein mann kocht in der küche",
        "eine frau tanzt im zimmer",
        "ein kind spielt mit spielzeug",
        "zwei menschen gehen zusammen",
        "drei hunde spielen draußen",
        "vier vögel sitzen auf einem baum",
        "fünf kinder laufen auf dem feld",
        "sechs katzen schlafen auf dem bett",
        "sieben menschen stehen in einer reihe",
        "acht blumen blühen im garten",
        "neun autos fahren auf der straße",
        "zehn studenten lernen in der bibliothek",
        "viele vögel fliegen am himmel",
        "die sonne scheint hell",
        "der mond ist voll und hell",
        "die sterne funkeln in der nacht",
        "der regen fällt sanft",
        "der schnee fällt sanft",
        "der wind weht stark",
        "das wasser fließt sanft",
        "das feuer brennt heiß",
        "der baum wächst hoch",
        "die blume blüht schön",
        "der vogel singt laut",
    ]

    val_en = [
        "a man is walking alone",
        "a woman is singing a song",
        "a dog is barking loudly",
        "a cat is meowing softly",
        "a bird is chirping happily",
    ]

    val_de = [
        "ein mann geht allein",
        "eine frau singt ein lied",
        "ein hund bellt laut",
        "eine katze miaut sanft",
        "ein vogel zwitschert fröhlich",
    ]

    test_en = [
        "a person is smiling",
        "an animal is jumping",
        "a plant is growing",
        "the weather is nice",
        "the day is beautiful",
    ]

    test_de = [
        "eine person lächelt",
        "ein tier springt",
        "eine pflanze wächst",
        "das wetter ist schön",
        "der tag ist schön",
    ]

    def write_file(filepath, lines):
        with open(filepath, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")

    write_file(os.path.join(data_dir, "train.en"), train_en)
    write_file(os.path.join(data_dir, "train.de"), train_de)
    write_file(os.path.join(data_dir, "val.en"), val_en)
    write_file(os.path.join(data_dir, "val.de"), val_de)
    write_file(os.path.join(data_dir, "test.en"), test_en)
    write_file(os.path.join(data_dir, "test.de"), test_de)

    print(f"测试数据集已创建在: {data_dir}")
    print(f"  训练集: {len(train_en)} 对句子")
    print(f"  验证集: {len(val_en)} 对句子")
    print(f"  测试集: {len(test_en)} 对句子")
    print("\n使用测试数据集训练:")
    print("  python train_translation.py --data_dir data/test_translation --epochs 50")
    print("=" * 60)


if __name__ == "__main__":
    create_test_data()