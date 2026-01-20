# Candidate Elimination Algorithm - SIMPLE & WORKING

def candidate_elimination():

    X = [
        ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same'],
        ['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same'],
        ['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change'],
        ['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change']
    ]

    Y = ['Yes', 'Yes', 'No', 'Yes']

    # Initialize S and G
    S = ['0'] * len(X[0])
    G = ['?'] * len(X[0])

    print("Initial S:", S)
    print("Initial G:", G)

    for i in range(len(X)):
        if Y[i] == 'Yes':
            for j in range(len(X[0])):
                if S[j] == '0':
                    S[j] = X[i][j]
                elif S[j] != X[i][j]:
                    S[j] = '?'
            print(f"\nAfter positive example {i+1}")
            print("S:", S)
        else:
            for j in range(len(X[0])):
                if S[j] != X[i][j]:
                    G[j] = S[j]
            print(f"\nAfter negative example {i+1}")
            print("G:", G)

    print("\nFinal Specific Hypothesis:", S)
    print("Final General Hypothesis:", G)


# MAIN
candidate_elimination()

