#include <metal_stdlib>
using namespace metal;

#define TILE_SIZE 8

#define TILE_MAT1_ROWS 64
#define TILE_MAT1_COLS 32
#define TILE_MAT2_COLS 64
#define COARSE_FACTOR_COL 4
#define COARSE_FACTOR_ROW 4
#define THREADS_PER_BLOCK 256
#define THREADGROUP_WIDTH 16

#define LOAD_MAT1( \
    mat1, mat1Tile, group_id, mat1Rows, mat1Cols, \
    mat1TileStartCol, mat1TileStartRow, mat1Stride, t \
) { \
    for (uint i = 0; i < TILE_MAT1_ROWS; i += mat1Stride) { \
        uint mat1Row = group_id.y * TILE_MAT1_ROWS + i + mat1TileStartRow; \
        uint mat1Col = t * TILE_MAT1_COLS + mat1TileStartCol*4; \
        uint baseIdx = mat1Row * mat1Cols + mat1Col; \
        \
        if (mat1Row < mat1Rows && mat1Col + 3 < mat1Cols) { \
            float4 mat1Temp = *(device const float4*)&mat1[baseIdx]; \
            mat1Tile[mat1TileStartCol * 4 + 0][i + mat1TileStartRow] = mat1Temp[0]; \
            mat1Tile[mat1TileStartCol * 4 + 1][i + mat1TileStartRow] = mat1Temp[1]; \
            mat1Tile[mat1TileStartCol * 4 + 2][i + mat1TileStartRow] = mat1Temp[2]; \
            mat1Tile[mat1TileStartCol * 4 + 3][i + mat1TileStartRow] = mat1Temp[3]; \
        \
        } else { \
            mat1Tile[mat1TileStartCol * 4 + 0][i + mat1TileStartRow] = \
                (mat1Row < mat1Rows && mat1Col + 0 < mat1Cols) ? mat1[baseIdx + 0] : 0.0f; \
            \
            mat1Tile[mat1TileStartCol * 4 + 1][i + mat1TileStartRow] = \
                (mat1Row < mat1Rows && mat1Col + 1 < mat1Cols) ? mat1[baseIdx + 1] : 0.0f; \
            \
            mat1Tile[mat1TileStartCol * 4 + 2][i + mat1TileStartRow] = \
                (mat1Row < mat1Rows && mat1Col + 2 < mat1Cols) ? mat1[baseIdx + 2] : 0.0f; \
            \
            mat1Tile[mat1TileStartCol * 4 + 3][i + mat1TileStartRow] = 0.0f; \
        } \
    } \
}

#define LOAD_MAT2( \
    mat2, mat2Tile, group_id, mat1Cols, mat2Cols, \
    mat2TileStartCol, mat2TileStartRow, mat2Stride, t \
) { \
    for (uint i = 0; i < TILE_MAT1_COLS; i += mat2Stride) { \
        uint mat2Row = t * TILE_MAT1_COLS + i + mat2TileStartRow; \
        uint mat2Col = group_id.x * TILE_MAT2_COLS + mat2TileStartCol*4; \
        uint baseIdx = mat2Row * mat2Cols + mat2Col; \
        \
        if (mat2Row < mat1Cols && mat2Col + 3 < mat2Cols) { \
            float4 mat2Temp = *(device const float4*)&mat2[baseIdx]; \
            mat2Tile[i + mat2TileStartRow][mat2TileStartCol * 4 + 0] = mat2Temp[0]; \
            mat2Tile[i + mat2TileStartRow][mat2TileStartCol * 4 + 1] = mat2Temp[1]; \
            mat2Tile[i + mat2TileStartRow][mat2TileStartCol * 4 + 2] = mat2Temp[2]; \
            mat2Tile[i + mat2TileStartRow][mat2TileStartCol * 4 + 3] = mat2Temp[3]; \
        \
        } else { \
            mat2Tile[i + mat2TileStartRow][mat2TileStartCol * 4 + 0] = \
                (mat2Row < mat1Cols && mat2Col + 0 < mat2Cols) ? mat2[baseIdx + 0] : 0.0f; \
            \
            mat2Tile[i + mat2TileStartRow][mat2TileStartCol * 4 + 1] = \
                (mat2Row < mat1Cols && mat2Col + 1 < mat2Cols) ? mat2[baseIdx + 1] : 0.0f; \
            \
            mat2Tile[i + mat2TileStartRow][mat2TileStartCol * 4 + 2] = \
                (mat2Row < mat1Cols && mat2Col + 2 < mat2Cols) ? mat2[baseIdx + 2] : 0.0f; \
            \
            mat2Tile[i + mat2TileStartRow][mat2TileStartCol * 4 + 3] = 0.0f; \
        } \
    } \
}

#define LOAD_MAT2_T( \
    mat2, mat2Tile, mat1Cols, mat2Cols, \
    mat2TileRow, mat2TileCol, \
    float4PerThread, threadNum, group_id, t \
) { \
    for (uint i = 0; i < float4PerThread; i++) { \
        uint flatTile2Idx = threadNum + THREADS_PER_BLOCK * i; \
        \
        uint mat2TileRow = flatTile2Idx / (TILE_MAT1_COLS/4); \
        uint mat2TileCol = (flatTile2Idx % (TILE_MAT1_COLS/4)) * 4; \
        \
        uint mat2Row = group_id.x * TILE_MAT2_COLS  + mat2TileRow; \
        uint mat2Col = t * TILE_MAT1_COLS + mat2TileCol; \
        uint baseIdx = mat2Row * mat1Cols + mat2Col; \
        \
        if (mat2Row < mat2Cols && mat2Col + 3 < mat1Cols) { \
            float4 mat2Temp = *(device const float4*)&mat2[baseIdx]; \
            \
            mat2Tile[mat2TileCol + 0][mat2TileRow] = mat2Temp[0]; \
            mat2Tile[mat2TileCol + 1][mat2TileRow] = mat2Temp[1]; \
            mat2Tile[mat2TileCol + 2][mat2TileRow] = mat2Temp[2]; \
            mat2Tile[mat2TileCol + 3][mat2TileRow] = mat2Temp[3]; \
        } else { \
            mat2Tile[mat2TileCol + 0][mat2TileRow] = \
                (mat2Row < mat2Cols && mat2Col + 0 < mat1Cols) ? mat2[baseIdx + 0] : 0.0f; \
            \
            mat2Tile[mat2TileCol + 1][mat2TileRow] = \
                (mat2Row < mat2Cols && mat2Col + 1 < mat1Cols) ? mat2[baseIdx + 1] : 0.0f; \
            \
            mat2Tile[mat2TileCol + 2][mat2TileRow] = \
                (mat2Row < mat2Cols && mat2Col + 2 < mat1Cols) ? mat2[baseIdx + 2] : 0.0f; \
            \
            mat2Tile[mat2TileCol + 3][mat2TileRow] = 0.0f; \
        } \
    } \
}


#define LOAD_MAT1_T( \
    mat1, mat1Rows, mat1Cols, \
    mat1TileStartRow, mat1TileStartCol, group_id, t \
) { \
    for (uint i = 0; i < TILE_MAT1_COLS; i += mat1Stride) { \
        uint mat1TileRow = i + mat1TileStartRow; \
        uint mat1TileCol = mat1TileStartCol; \
        \
        uint mat1Row = t * TILE_MAT1_COLS + mat1TileRow; \
        uint mat1Col = group_id.y * TILE_MAT1_ROWS + mat1TileCol; \
        uint baseIdx = mat1Row * mat1Rows + mat1Col; \
        \
        if (mat1Row < mat1Cols && mat1Col + 3 < mat1Rows) { \
            float4 mat1Temp = *(device const float4*)&mat1[baseIdx]; \
            mat1Tile[mat1TileRow][mat1TileCol + 0] = mat1Temp[0]; \
            mat1Tile[mat1TileRow][mat1TileCol + 1] = mat1Temp[1]; \
            mat1Tile[mat1TileRow][mat1TileCol + 2] = mat1Temp[2]; \
            mat1Tile[mat1TileRow][mat1TileCol + 3] = mat1Temp[3]; \
        \
        } else { \
            mat1Tile[mat1TileRow][mat1TileCol + 0] = \
                (mat1Row < mat1Cols && mat1Col + 0 < mat1Rows) ? mat1[baseIdx + 0] : 0.0f; \
            \
            mat1Tile[mat1TileRow][mat1TileCol + 1] = \
                (mat1Row < mat1Cols && mat1Col + 1 < mat1Rows) ? mat1[baseIdx + 1] : 0.0f; \
            \
            mat1Tile[mat1TileRow][mat1TileCol + 2] = \
                (mat1Row < mat1Cols && mat1Col + 2 < mat1Rows) ? mat1[baseIdx + 2] : 0.0f; \
            \
            mat1Tile[mat1TileRow][mat1TileCol + 3]= 0.0f; \
        } \
    } \
}

#define LOAD_REGISTERS(regMat1, mat1Tile, regMat2, mat2Tile, prodVals) { \
    for (uint k = 0; k < TILE_MAT1_COLS; k++) { \
        for (uint i = 0; i < COARSE_FACTOR_ROW; i++) { \
            regMat1[i] = mat1Tile[k][row + i]; \
        } \
        \
        for (uint i = 0; i < COARSE_FACTOR_COL; i++) { \
            regMat2[i] = mat2Tile[k][col + i]; \
        } \
        \
        for (uint i = 0; i < COARSE_FACTOR_ROW; i++) { \
            for (uint j = 0; j < COARSE_FACTOR_COL; j++) { \
                prodVals[i][j] += regMat1[i] * regMat2[j]; \
            } \
        } \
    } \
}

#define MM_LOAD_DATA( \
        mat1, mat2, \
        mat1Tile, mat2Tile, \
        prodVals, regMat1, regMat2, \
        dims, threadNum, row, col, group_id \
){ \
    uint mat1Rows = dims[0]; \
    uint mat1Cols = dims[1]; \
    uint mat2Cols = dims[2]; \
    \
    uint numTiles = (mat1Cols + TILE_MAT1_COLS - 1)/TILE_MAT1_COLS; \
    \
    uint mat1TileStartRow = threadNum / (TILE_MAT1_COLS / 4); \
    uint mat1TileStartCol = threadNum % (TILE_MAT1_COLS / 4); \
    uint mat1Stride = THREADS_PER_BLOCK / (TILE_MAT1_COLS / 4); \
    \
    uint mat2TileStartRow = threadNum / (TILE_MAT2_COLS / 4); \
    uint mat2TileStartCol = threadNum % (TILE_MAT2_COLS / 4); \
    uint mat2Stride = THREADS_PER_BLOCK / (TILE_MAT2_COLS / 4); \
    \
    for (uint t = 0; t < numTiles; t++) { \
        LOAD_MAT1( \
            mat1, mat1Tile, group_id, mat1Rows, mat1Cols, \
            mat1TileStartCol, mat1TileStartRow, mat1Stride, t \
        ) \
        \
        LOAD_MAT2( \
            mat2, mat2Tile, group_id, mat1Cols, mat2Cols, \
            mat2TileStartCol, mat2TileStartRow, mat2Stride, t \
        ) \
        \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
        \
        LOAD_REGISTERS(regMat1, mat1Tile, regMat2, mat2Tile, prodVals) \
        \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
    } \
} 

#define MMT_LOAD_DATA( \
    mat1, mat2, \
    mat1Tile, mat2Tile, \
    prodVals, regMat1, regMat2, \
    dims, threadNum, row, col, group_id \
) { \
    uint mat1Rows = dims[0]; \
    uint mat1Cols = dims[1]; \
    uint mat2Cols = dims[2]; \
    \
    uint float4PerThread = ((TILE_MAT1_COLS * TILE_MAT2_COLS) / 4) / THREADS_PER_BLOCK; \
    \
    uint numTiles = (mat1Cols + TILE_MAT1_COLS - 1)/TILE_MAT1_COLS; \
    \
    uint mat1TileStartRow = threadNum / (TILE_MAT1_COLS / 4); \
    uint mat1TileStartCol = threadNum % (TILE_MAT1_COLS / 4); \
    uint mat1Stride = THREADS_PER_BLOCK / (TILE_MAT1_COLS / 4); \
    \
    for (uint t = 0; t < numTiles; t++) { \
        LOAD_MAT1( \
            mat1, mat1Tile, group_id, mat1Rows, mat1Cols, \
            mat1TileStartCol, mat1TileStartRow, mat1Stride, t \
        ) \
        \
        LOAD_MAT2_T( \
            mat2, mat2Tile, mat1Cols, mat2Cols, \
            mat2TileRow, mat2TileCol, \
            float4PerThread, threadNum, group_id, t \
        ) \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
        \
        LOAD_REGISTERS(regMat1, mat1Tile, regMat2, mat2Tile, prodVals) \
        \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
    } \
}

#define MTM_LOAD_DATA( \
    mat1, mat2, \
    mat1Tile, mat2Tile, \
    prodVals, regMat1, regMat2, \
    dims, threadNum, row, col, group_id \
) { \
    uint mat1Rows = dims[0]; \
    uint mat1Cols = dims[1]; \
    uint mat2Cols = dims[2]; \
    \
    uint numTiles = (mat1Cols + TILE_MAT1_COLS - 1)/TILE_MAT1_COLS; \
    \
    uint mat1TileStartRow = threadNum / (TILE_MAT1_ROWS / 4); \
    uint mat1TileStartCol = (threadNum % (TILE_MAT1_ROWS / 4)) * 4; \
    uint mat1Stride = THREADS_PER_BLOCK / (TILE_MAT1_ROWS  / 4); \
    \
    uint mat2TileStartRow = threadNum / (TILE_MAT2_COLS / 4); \
    uint mat2TileStartCol = threadNum % (TILE_MAT2_COLS / 4); \
    uint mat2Stride = THREADS_PER_BLOCK / (TILE_MAT2_COLS / 4); \
    \
    for (uint t = 0; t < numTiles; t++) { \
        LOAD_MAT1_T( \
            mat1, mat1Rows, mat1Cols, \
            mat1TileStartRow, mat1TileStartCol, group_id, t \
        ) \
        \
        LOAD_MAT2( \
            mat2, mat2Tile, group_id, mat1Cols, mat2Cols, \
            mat2TileStartCol, mat2TileStartRow, mat2Stride, t \
        ) \
        \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
        \
        LOAD_REGISTERS(regMat1, mat1Tile, regMat2, mat2Tile, prodVals) \
        \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
    } \
}

#define MTMT_LOAD_DATA( \
    mat1, mat2, \
    mat1Tile, mat2Tile, \
    prodVals, regMat1, regMat2, \
    dims, threadNum, row, col, group_id \
) { \
    uint mat1Rows = dims[0]; \
    uint mat1Cols = dims[1]; \
    uint mat2Cols = dims[2]; \
    \
    uint float4PerThread = ((TILE_MAT1_COLS * TILE_MAT2_COLS) / 4) / THREADS_PER_BLOCK; \
    \
    uint numTiles = (mat1Cols + TILE_MAT1_COLS - 1)/TILE_MAT1_COLS; \
    \
    uint mat1TileStartRow = threadNum / (TILE_MAT1_ROWS / 4); \
    uint mat1TileStartCol = (threadNum % (TILE_MAT1_ROWS / 4)) * 4; \
    uint mat1Stride = THREADS_PER_BLOCK / (TILE_MAT1_ROWS  / 4); \
    \
    for (uint t = 0; t < numTiles; t++) { \
        LOAD_MAT1_T( \
            mat1, mat1Rows, mat1Cols, \
            mat1TileStartRow, mat1TileStartCol, group_id, t \
        ) \
        \
        LOAD_MAT2_T( \
            mat2, mat2Tile, mat1Cols, mat2Cols, \
            mat2TileRow, mat2TileCol, \
            float4PerThread, threadNum, group_id, t \
        ) \
        \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
        \
        LOAD_REGISTERS(regMat1, mat1Tile, regMat2, mat2Tile, prodVals) \
        \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
    } \
}

kernel void mm(
    device const float *mat1 [[ buffer(0) ]],
    device const float *mat2 [[ buffer(1) ]],
    device float *prodMat [[ buffer(2) ]],
    constant uint3 &dims [[ buffer(3) ]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 group_id [[ threadgroup_position_in_grid ]]
) {
    uint mat1Rows = dims[0];
    uint mat2Cols = dims[2];

    uint threadNum = tid.y * THREADGROUP_WIDTH + tid.x;

    uint row = COARSE_FACTOR_ROW * (threadNum / (TILE_MAT2_COLS/COARSE_FACTOR_COL));
    uint col = COARSE_FACTOR_COL * (threadNum % (TILE_MAT2_COLS/COARSE_FACTOR_COL));
    
    threadgroup float mat1Tile[TILE_MAT1_COLS][TILE_MAT1_ROWS];
    threadgroup float mat2Tile[TILE_MAT1_COLS][TILE_MAT2_COLS];

    float prodVals[COARSE_FACTOR_ROW][COARSE_FACTOR_COL] = {{0.0f}};
    float regMat1[COARSE_FACTOR_ROW] = {0.0f};
    float regMat2[COARSE_FACTOR_COL] = {0.0f};

    MM_LOAD_DATA(
        mat1, mat2, mat1Tile, mat2Tile, prodVals, regMat1, 
        regMat2, dims, threadNum, row, col, group_id
    );

    for (uint i = 0; i < COARSE_FACTOR_ROW; i++) {
        for (uint j = 0; j < COARSE_FACTOR_COL; j++) {
            uint prodRow = group_id.y * TILE_MAT1_ROWS + row + i;
            uint prodCol = group_id.x * TILE_MAT2_COLS + col + j;

            if (prodRow < mat1Rows && prodCol < mat2Cols) {
                prodMat[prodRow * mat2Cols + prodCol] = prodVals[i][j];
            }
        }
    }
}

kernel void mmBiasReLU(
    device const float *mat1 [[ buffer(0) ]],
    device const float *mat2 [[ buffer(1) ]],
    device float *a [[ buffer(2) ]],
    device const float *biases [[ buffer(3) ]],
    constant uint3 &dims [[ buffer(4) ]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 group_id [[ threadgroup_position_in_grid ]]
) {
    uint mat1Rows = dims[0];
    uint mat2Cols = dims[2];

    uint threadNum = tid.y * THREADGROUP_WIDTH + tid.x;

    uint row = COARSE_FACTOR_ROW * (threadNum / (TILE_MAT2_COLS/COARSE_FACTOR_COL));
    uint col = COARSE_FACTOR_COL * (threadNum % (TILE_MAT2_COLS/COARSE_FACTOR_COL));
    
    threadgroup float mat1Tile[TILE_MAT1_COLS][TILE_MAT1_ROWS];
    threadgroup float mat2Tile[TILE_MAT1_COLS][TILE_MAT2_COLS];

    float prodVals[COARSE_FACTOR_ROW][COARSE_FACTOR_COL] = {{0.0f}};
    float regMat1[COARSE_FACTOR_ROW] = {0.0f};
    float regMat2[COARSE_FACTOR_COL] = {0.0f};
    float regBias[COARSE_FACTOR_COL];
    
    for (uint j = 0; j < COARSE_FACTOR_COL; j++) {
        uint prodCol = group_id.x * TILE_MAT2_COLS + col + j;
        regBias[j] = (prodCol < mat2Cols) ? biases[prodCol] : 0.0f;
    }

    MM_LOAD_DATA(
        mat1, mat2, mat1Tile, mat2Tile, prodVals, regMat1, 
        regMat2, dims, threadNum, row, col, group_id
    );

    for (uint i = 0; i < COARSE_FACTOR_ROW; i++) {
        for (uint j = 0; j < COARSE_FACTOR_COL; j++) {
            uint prodRow = group_id.y * TILE_MAT1_ROWS + row + i;
            uint prodCol = group_id.x * TILE_MAT2_COLS + col + j;

            if (prodRow < mat1Rows && prodCol < mat2Cols) {
                a[prodRow * mat2Cols + prodCol] = max(prodVals[i][j] + regBias[j], 0.0f);
            }
        }
    }
}

kernel void mmTBiasReLU(
    device const float *mat1 [[ buffer(0) ]],
    device const float *mat2 [[ buffer(1) ]],
    device float *a [[ buffer(2) ]],
    device const float *biases [[ buffer(3) ]],
    constant uint3 &dims [[ buffer(4) ]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 group_id [[ threadgroup_position_in_grid ]]
) {
    uint mat1Rows = dims[0];
    uint mat2Cols = dims[2];

    uint threadNum = tid.y * THREADGROUP_WIDTH + tid.x;

    uint row = COARSE_FACTOR_ROW * (threadNum / (TILE_MAT2_COLS/COARSE_FACTOR_COL));
    uint col = COARSE_FACTOR_COL * (threadNum % (TILE_MAT2_COLS/COARSE_FACTOR_COL));

    threadgroup float mat1Tile[TILE_MAT1_COLS][TILE_MAT1_ROWS];
    threadgroup float mat2Tile[TILE_MAT1_COLS][TILE_MAT2_COLS];

    float prodVals[COARSE_FACTOR_ROW][COARSE_FACTOR_COL] = {{0.0f}};
    float regMat1[COARSE_FACTOR_ROW] = {0.0f};
    float regMat2[COARSE_FACTOR_COL] = {0.0f};
    float regBias[COARSE_FACTOR_COL];
    
    for (uint j = 0; j < COARSE_FACTOR_COL; j++) {
        uint prodCol = group_id.x * TILE_MAT2_COLS + col + j;
        regBias[j] = (prodCol < mat2Cols) ? biases[prodCol] : 0.0f;
    }

    MMT_LOAD_DATA( 
        mat1, mat2, 
        mat1Tile, mat2Tile, 
        prodVals, regMat1, regMat2, 
        dims, threadNum, row, col, group_id 
    );

    for (uint i = 0; i < COARSE_FACTOR_ROW; i++) {
        for (uint j = 0; j < COARSE_FACTOR_COL; j++) {
            uint prodRow = group_id.y * TILE_MAT1_ROWS + row + i;
            uint prodCol = group_id.x * TILE_MAT2_COLS + col + j;

            if (prodRow < mat1Rows && prodCol < mat2Cols) {
                a[prodRow * mat2Cols + prodCol] = max(prodVals[i][j] + regBias[j], 0.0f);
            }
        }
    }
}

kernel void mmT(
    device const float *mat1 [[ buffer(0) ]],
    device const float *mat2 [[ buffer(1) ]],
    device float *prodMat [[ buffer(2) ]],
    constant uint3 &dims [[ buffer(3) ]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 gid [[thread_position_in_grid]],
    uint2 group_id [[ threadgroup_position_in_grid ]]
) {
    uint mat1Rows = dims[0];
    uint mat2Cols = dims[2];

    uint threadNum = tid.y * THREADGROUP_WIDTH + tid.x;

    uint row = COARSE_FACTOR_ROW * (threadNum / (TILE_MAT2_COLS/COARSE_FACTOR_COL));
    uint col = COARSE_FACTOR_COL * (threadNum % (TILE_MAT2_COLS/COARSE_FACTOR_COL));

    threadgroup float mat1Tile[TILE_MAT1_COLS][TILE_MAT1_ROWS];
    threadgroup float mat2Tile[TILE_MAT1_COLS][TILE_MAT2_COLS];

    float prodVals[COARSE_FACTOR_ROW][COARSE_FACTOR_COL] = {{0.0f}};
    float regMat1[COARSE_FACTOR_ROW] = {0.0f};
    float regMat2[COARSE_FACTOR_COL] = {0.0f};

    MMT_LOAD_DATA( 
        mat1, mat2, 
        mat1Tile, mat2Tile, 
        prodVals, regMat1, regMat2, 
        dims, threadNum, row, col, group_id 
    );

    for (uint i = 0; i < COARSE_FACTOR_ROW; i++) {
        for (uint j = 0; j < COARSE_FACTOR_COL; j++) {
            uint prodRow = group_id.y * TILE_MAT1_ROWS + row + i;
            uint prodCol = group_id.x * TILE_MAT2_COLS + col + j;

            if (prodRow < mat1Rows && prodCol < mat2Cols) {
                prodMat[prodRow * mat2Cols + prodCol] = prodVals[i][j];
            }
        }
    }
}

kernel void mTm(
    device const float *mat1 [[ buffer(0) ]],
    device const float *mat2 [[ buffer(1) ]],
    device float *prodMat [[ buffer(2) ]],
    constant uint3 &dims [[ buffer(3) ]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 gid [[thread_position_in_grid]],
    uint2 group_id [[ threadgroup_position_in_grid ]]
) {
    uint mat1Rows = dims[0];
    uint mat2Cols = dims[2];

    uint threadNum = tid.y * THREADGROUP_WIDTH + tid.x;

    uint row = COARSE_FACTOR_ROW * (threadNum / (TILE_MAT2_COLS/COARSE_FACTOR_COL));
    uint col = COARSE_FACTOR_COL * (threadNum % (TILE_MAT2_COLS/COARSE_FACTOR_COL));

    threadgroup float mat1Tile[TILE_MAT1_COLS][TILE_MAT1_ROWS];
    threadgroup float mat2Tile[TILE_MAT1_COLS][TILE_MAT2_COLS];

    float prodVals[COARSE_FACTOR_ROW][COARSE_FACTOR_COL] = {{0.0f}};
    float regMat1[COARSE_FACTOR_ROW] = {0.0f};
    float regMat2[COARSE_FACTOR_COL] = {0.0f};

    MTM_LOAD_DATA(
        mat1, mat2, 
        mat1Tile, mat2Tile, 
        prodVals, regMat1, regMat2, 
        dims, threadNum, row, col, group_id 
    );

    for (uint i = 0; i < COARSE_FACTOR_ROW; i++) {
        for (uint j = 0; j < COARSE_FACTOR_COL; j++) {
            uint prodRow = group_id.y * TILE_MAT1_ROWS + row + i;
            uint prodCol = group_id.x * TILE_MAT2_COLS + col + j;

            if (prodRow < mat1Rows && prodCol < mat2Cols) {
                prodMat[prodRow * mat2Cols + prodCol] = prodVals[i][j];
            }
        }
    }
}

kernel void applyWeightsGrad(
    device const float *mat1 [[ buffer(0) ]],
    device const float *mat2 [[ buffer(1) ]],
    device float *weights [[ buffer(2) ]],
    constant uint3 &dims [[ buffer(3) ]],
    constant float &scaleFactor [[ buffer(4) ]],
    constant float &weightL2 [[ buffer(5) ]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 gid [[thread_position_in_grid]],
    uint2 group_id [[ threadgroup_position_in_grid ]]
) {
    uint mat1Rows = dims[0];
    uint mat2Cols = dims[2];

    uint threadNum = tid.y * THREADGROUP_WIDTH + tid.x;

    uint row = COARSE_FACTOR_ROW * (threadNum / (TILE_MAT2_COLS/COARSE_FACTOR_COL));
    uint col = COARSE_FACTOR_COL * (threadNum % (TILE_MAT2_COLS/COARSE_FACTOR_COL));

    threadgroup float mat1Tile[TILE_MAT1_COLS][TILE_MAT1_ROWS];
    threadgroup float mat2Tile[TILE_MAT1_COLS][TILE_MAT2_COLS];

    float prodVals[COARSE_FACTOR_ROW][COARSE_FACTOR_COL] = {{0.0f}};
    float regMat1[COARSE_FACTOR_ROW] = {0.0f};
    float regMat2[COARSE_FACTOR_COL] = {0.0f};

    float l2Term = 2 * weightL2;

    MTM_LOAD_DATA(
        mat1, mat2, 
        mat1Tile, mat2Tile, 
        prodVals, regMat1, regMat2, 
        dims, threadNum, row, col, group_id 
    );

    for (uint i = 0; i < COARSE_FACTOR_ROW; i++) {
        for (uint j = 0; j < COARSE_FACTOR_COL; j++) {
            uint prodRow = group_id.y * TILE_MAT1_ROWS + row + i;
            uint prodCol = group_id.x * TILE_MAT2_COLS + col + j;

            if (prodRow < mat1Rows && prodCol < mat2Cols) {
                uint idx = prodRow * mat2Cols + prodCol;
                float w = weights[idx];
                w += scaleFactor * (prodVals[i][j] + l2Term * w);
                weights[idx] = w;
            }
        }
    }
}

kernel void mTmT(
    device const float *mat1 [[ buffer(0) ]],
    device const float *mat2 [[ buffer(1) ]],
    device float *prodMat [[ buffer(2) ]],
    constant uint3 &dims [[ buffer(3) ]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 gid [[thread_position_in_grid]],
    uint2 group_id [[ threadgroup_position_in_grid ]]
) {
    uint mat1Rows = dims[0];
    uint mat2Cols = dims[2];

    uint threadNum = tid.y * THREADGROUP_WIDTH + tid.x;

    uint row = COARSE_FACTOR_ROW * (threadNum / (TILE_MAT2_COLS/COARSE_FACTOR_COL));
    uint col = COARSE_FACTOR_COL * (threadNum % (TILE_MAT2_COLS/COARSE_FACTOR_COL));

    threadgroup float mat1Tile[TILE_MAT1_COLS][TILE_MAT1_ROWS];
    threadgroup float mat2Tile[TILE_MAT1_COLS][TILE_MAT2_COLS];

    float prodVals[COARSE_FACTOR_ROW][COARSE_FACTOR_COL] = {{0.0f}};
    float regMat1[COARSE_FACTOR_ROW] = {0.0f};
    float regMat2[COARSE_FACTOR_COL] = {0.0f};

    MTMT_LOAD_DATA(
        mat1, mat2, 
        mat1Tile, mat2Tile, 
        prodVals, regMat1, regMat2, 
        dims, threadNum, row, col, group_id 
    );

    for (uint i = 0; i < COARSE_FACTOR_ROW; i++) {
        for (uint j = 0; j < COARSE_FACTOR_COL; j++) {
            uint prodRow = group_id.y * TILE_MAT1_ROWS + row + i;
            uint prodCol = group_id.x * TILE_MAT2_COLS + col + j;

            if (prodRow < mat1Rows && prodCol < mat2Cols) {
                prodMat[prodRow * mat2Cols + prodCol] = prodVals[i][j];
            }
        }
    }
}

kernel void colSums(
    device const float *mat [[ buffer(0) ]],
    device float *vec [[ buffer (1) ]],
    constant uint *dims [[ buffer(2) ]],
    uint gid [[thread_position_in_grid]]
) {
    uint col = gid;

    uint matRows = dims[0];
    uint matCols = dims[1];

    if (col >= matCols)
        return;

    float val = 0.0;
    for (uint i = 0; i < matRows; i++) {
        val += mat[i * matCols + col];
    }

    vec[col] = val;
}

kernel void applyBiasGradDense(
    device const float *grad[[ buffer(0) ]],
    device float *biases [[ buffer (1) ]],
    constant uint2 &dims [[ buffer(2) ]],
    constant float &scaleFactor [[ buffer(3) ]],
    uint gid [[thread_position_in_grid]]
) {
    uint col = gid;

    uint gradRows = dims[0];
    uint gradCols = dims[1];

    if (col >= gradCols)
        return;

    float val = 0.0;
    for (uint i = 0; i < gradRows; i++) {
        val += grad[i * gradCols + col];
    }

    biases[col] += (val * scaleFactor);
}

kernel void addToRows(
    device float *mat [[ buffer(0) ]],
    device const float *vec [[ buffer (1) ]],
    constant uint *dims [[ buffer(2) ]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint i = gid.y;
    uint j = gid.x;

    uint matRows = dims[0];
    uint matCols = dims[1];

    if (i >= matRows || j >= matCols)
        return;

    mat[i * matCols + j] += vec[j];
}