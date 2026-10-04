# =============================================
# 역행렬 계산 프로그램
# 방법 1: 행렬식(Determinant)을 이용한 역행렬
# 방법 2: 가우스-조던 소거법을 이용한 역행렬
# =============================================


# ──────────────────────────────────────────────
# 1. 행렬 입력 기능
#    사용자로부터 정수 n을 입력받아
#    n × n 크기의 정방행렬을 행 단위로 입력받고
#    2차원 리스트로 저장
# ──────────────────────────────────────────────
def input_matrix():
    n = int(input("행렬의 크기 n을 입력하세요: "))
    print(f"{n}x{n} 행렬의 각 행을 공백으로 구분하여 입력하세요:")
    matrix = []
    for i in range(n):
        row = list(map(float, input(f"  {i+1}행: ").split()))
        if len(row) != n:
            raise ValueError(f"입력 오류: {n}개의 값을 입력해야 합니다.")
        matrix.append(row)
    return n, matrix


# ──────────────────────────────────────────────
# 2. 행렬식을 이용한 역행렬 계산 기능
# ──────────────────────────────────────────────

# 소행렬(Minor matrix)을 구하는 함수
# i행, j열을 제거한 (n-1)×(n-1) 부분행렬 반환
def get_minor(matrix, i, j):
    return [
        [matrix[r][c] for c in range(len(matrix)) if c != j]
        for r in range(len(matrix)) if r != i
    ]


# 행렬식(Determinant)을 재귀적으로 계산하는 함수
# 1×1 행렬: 원소 자체가 행렬식
# 2×2 행렬: ad - bc
# n×n 행렬: 첫 번째 행 기준 여인수 전개(cofactor expansion)
def determinant(matrix):
    n = len(matrix)
    if n == 1:
        return matrix[0][0]
    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    det = 0
    for j in range(n):
        # 여인수(cofactor) = (-1)^(i+j) * M_ij
        cofactor = ((-1) ** j) * determinant(get_minor(matrix, 0, j))
        det += matrix[0][j] * cofactor
    return det


# 여인수 행렬(Cofactor matrix)을 구하는 함수
# 각 원소 C_ij = (-1)^(i+j) * det(M_ij)
def cofactor_matrix(matrix):
    n = len(matrix)
    cof = []
    for i in range(n):
        row = []
        for j in range(n):
            minor = get_minor(matrix, i, j)
            row.append(((-1) ** (i + j)) * determinant(minor))
        cof.append(row)
    return cof


# 전치행렬(Transpose)을 구하는 함수
# 수반행렬(Adjugate) = 여인수 행렬의 전치
def transpose(matrix):
    n = len(matrix)
    return [[matrix[j][i] for j in range(n)] for i in range(n)]


# 행렬식 방법으로 역행렬을 계산하는 함수
# 역행렬 = (1/det) × adj(A)
# adj(A) = 여인수 행렬의 전치(수반행렬)
def inverse_by_determinant(matrix):
    n = len(matrix)
    det = determinant(matrix)

    # 행렬식이 0이면 역행렬이 존재하지 않음
    if det == 0:
        print("[오류] 행렬식이 0입니다. 역행렬이 존재하지 않습니다.")
        return None

    print(f"  행렬식(det) = {det}")

    # 수반행렬(adjugate) = 여인수 행렬의 전치
    adj = transpose(cofactor_matrix(matrix))

    # 역행렬 = (1/det) × adj(A)
    inverse = [
        [adj[i][j] / det for j in range(n)]
        for i in range(n)
    ]
    return inverse


# ──────────────────────────────────────────────
# 3. 가우스-조던 소거법을 이용한 역행렬 계산 기능
#    [A | I] 형태의 확대행렬(augmented matrix)을 만들고
#    행 연산을 통해 [I | A⁻¹] 형태로 변환
# ──────────────────────────────────────────────
def inverse_by_gauss_jordan(matrix):
    n = len(matrix)

    # 원본 행렬을 변경하지 않기 위해 깊은 복사
    # 확대행렬 [A | I] 생성 (n × 2n)
    aug = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(matrix)]

    for col in range(n):
        # 피벗 선택: 현재 열에서 절대값이 가장 큰 행을 피벗으로 선택 (부분 피벗팅)
        # 수치 안정성을 높이기 위한 기법
        max_row = col
        for row in range(col + 1, n):
            if abs(aug[row][col]) > abs(aug[max_row][col]):
                max_row = row

        # 피벗 행과 현재 행을 교환
        aug[col], aug[max_row] = aug[max_row], aug[col]

        # 피벗 원소가 0이면 역행렬이 존재하지 않음
        pivot = aug[col][col]
        if abs(pivot) < 1e-12:
            print("[오류] 피벗이 0입니다. 역행렬이 존재하지 않습니다.")
            return None

        # 피벗 행을 피벗 원소로 나누어 피벗을 1로 만듦
        for j in range(2 * n):
            aug[col][j] /= pivot

        # 피벗 열의 나머지 행을 모두 0으로 만듦 (행 뺄셈 연산)
        for row in range(n):
            if row != col:
                factor = aug[row][col]
                for j in range(2 * n):
                    aug[row][j] -= factor * aug[col][j]

    # 확대행렬의 오른쪽 절반이 역행렬
    inverse = [aug[i][n:] for i in range(n)]
    return inverse


# ──────────────────────────────────────────────
# 4. 결과 출력 및 비교 기능
# ──────────────────────────────────────────────

# 행렬을 보기 좋게 출력하는 함수
def print_matrix(matrix, label=""):
    if label:
        print(f"\n{label}:")
    for row in matrix:
        print("  [", "  ".join(f"{val:10.4f}" for val in row), "]")


# 두 행렬이 동일한지 비교하는 함수
# 부동소수점 오차를 고려하여 허용 오차(tolerance) 이내이면 동일하다고 판단
def matrices_equal(mat1, mat2, tol=1e-6):
    n = len(mat1)
    for i in range(n):
        for j in range(n):
            if abs(mat1[i][j] - mat2[i][j]) > tol:
                return False
    return True


# ──────────────────────────────────────────────
# 추가기능: A × A⁻¹ = I 역행렬 검증 기능
#   계산된 역행렬이 실제로 올바른지
#   원래 행렬과 곱하여 단위행렬이 되는지 확인
# ──────────────────────────────────────────────

# 행렬 곱셈 함수
def multiply_matrices(A, B):
    n = len(A)
    result = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                result[i][j] += A[i][k] * B[k][j]
    return result


# 단위행렬 생성 함수
def identity_matrix(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


# A × A⁻¹ = I 인지 검증하는 함수
# 원래 행렬과 역행렬을 곱한 결과가 단위행렬과 같은지 확인
def verify_inverse(matrix, inverse):
    n = len(matrix)
    product = multiply_matrices(matrix, inverse)
    identity = identity_matrix(n)
    return matrices_equal(product, identity)


# ──────────────────────────────────────────────
# 메인 실행부
# ──────────────────────────────────────────────
def main():
    # 1단계: 행렬 입력
    print("=" * 50)
    print("       역행렬 계산 프로그램")
    print("=" * 50)
    n, matrix = input_matrix()
    print_matrix(matrix, "입력된 행렬 A")

    # 2단계: 행렬식 방법으로 역행렬 계산
    print("\n" + "-" * 50)
    print("[방법 1] 행렬식을 이용한 역행렬 계산")
    print("-" * 50)
    inv_det = inverse_by_determinant(matrix)
    if inv_det:
        print_matrix(inv_det, "행렬식 방법 역행렬")

    # 3단계: 가우스-조던 소거법으로 역행렬 계산
    print("\n" + "-" * 50)
    print("[방법 2] 가우스-조던 소거법을 이용한 역행렬 계산")
    print("-" * 50)
    inv_gj = inverse_by_gauss_jordan(matrix)
    if inv_gj:
        print_matrix(inv_gj, "가우스-조던 방법 역행렬")

    # 4단계: 두 결과 비교
    print("\n" + "=" * 50)
    print("[결과 비교]")
    print("=" * 50)

    if inv_det is None and inv_gj is None:
        # 두 방법 모두 역행렬이 존재하지 않는 경우
        print("두 방법 모두 역행렬이 존재하지 않습니다. (특이행렬)")
    elif inv_det is None or inv_gj is None:
        # 한쪽만 실패한 경우
        print("한 방법에서만 역행렬을 구할 수 없었습니다. 결과가 불일치합니다.")
    elif matrices_equal(inv_det, inv_gj):
        # 두 결과가 허용 오차 이내로 동일한 경우
        print(">> 두 방법의 역행렬 계산 결과가 동일합니다. (허용 오차: 1e-6)")
    else:
        # 두 결과가 다른 경우
        print(">> 두 방법의 역행렬 계산 결과가 다릅니다.")
        print("   (부동소수점 연산 오차가 허용 범위를 초과했습니다)")

    # 5단계: 추가기능 - A × A⁻¹ = I 검증
    print("\n" + "=" * 50)
    print("[추가기능] A × A⁻¹ = I 역행렬 검증")
    print("=" * 50)

    if inv_det:
        product = multiply_matrices(matrix, inv_det)
        print_matrix(product, "A × A⁻¹ (행렬식 방법)")
        if verify_inverse(matrix, inv_det):
            print(">> 검증 성공: A × A⁻¹ = I (단위행렬)")
        else:
            print(">> 검증 실패: A × A⁻¹ ≠ I")
    else:
        print("역행렬이 존재하지 않아 검증을 수행할 수 없습니다.")

    print()


if __name__ == "__main__":
    main()
