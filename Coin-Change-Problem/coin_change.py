def coin_change(s : int, denominations : set[int] )  -> int:
    """
    Given a sum s and a list of denominations, 
    the algorithm finds the minimum number of coins needed to make the sum s. 
    If it is not possible to return number of coins,the algorithm returns -1.
    """

    if s < 0:
        return -1  

    if not denominations or any(coin <= 0 for coin in denominations):
        return -1
    
    if s == 0:
        return 0
    
    coin_change = [0] + [float('inf')] * s 

    for i in range(1, s + 1):

        for coin in denominations:
            if coin <= i:
                coin_change[i] = min(coin_change[i], coin_change[i - coin] + 1)

    return coin_change[s] if coin_change[s] != float('inf') else -1


def main():
    #Values provided in the assignment
    s             : int  = 1_040_528   
    denominations : set = {1, 5, 10, 20}

    print(f"Denominations: {denominations}")
    print(f"Sum: {s}")
    print(f"Minimum number of coins: {coin_change(s, denominations)}")


if __name__ == "__main__":
    main()
