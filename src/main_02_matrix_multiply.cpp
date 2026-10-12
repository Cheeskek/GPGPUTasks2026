#include <libbase/stats.h>
#include <libutils/misc.h>

#include <libbase/timer.h>
#include <libbase/fast_random.h>
#include <libgpu/vulkan/engine.h>
#include <libgpu/vulkan/tests/test_utils.h>

#include <libgpu/vulkan/vk/common_host.h>

#include "kernels/defines.h"
#include "kernels/kernels.h"

#include <fstream>
#include <iomanip>

namespace cpu {
void multiply(
    const std::vector<float> &a,
    const std::vector<float> &b,
          std::vector<float> &c,
                 unsigned int w,
                 unsigned int h,
                 unsigned int k,
                  bool with_omp)
{
    #pragma omp parallel for schedule(dynamic, 1) if (with_omp)
    for (ptrdiff_t j = 0; j < h; ++j) {
        for (ptrdiff_t i = 0; i < w; ++i) {
            float acc = 0.0f;

            for (int ki = 0; ki < k; ++ki) {
                acc += a[j * k + ki] * b[ki * w + i];
            }

            c[j * w + i] = acc;
        }
    }
}
}

namespace gpu {
struct gpu_half {
    uint16_t bits;
    gpu_half() = default;
    gpu_half(float x)  {
        // Reinterpret the float32 bit pattern (NOT a numeric conversion).
        uint32_t fltInt32;
        __builtin_memcpy(&fltInt32, &x, sizeof(fltInt32));

        uint16_t fltInt16  = static_cast<uint16_t>((fltInt32 >> 16) & 0x8000u); // sign
        uint32_t exponent  = (fltInt32 >> 23) & 0xffu;
        uint32_t mantissa  = fltInt32 & 0x7fffffu;

        if (exponent == 0xffu) {
            // Inf or NaN
            fltInt16 |= 0x7c00u;
            if (mantissa != 0u) {
                fltInt16 |= 0x0200u; // quiet NaN
            }
        } else {
            int32_t half_exp = static_cast<int32_t>(exponent) - 127 + 15;
            if (half_exp >= 0x1f) {
                // Overflow -> Inf
                fltInt16 |= 0x7c00u;
            } else if (half_exp <= 0) {
                // Subnormal half (or underflow to zero)
                if (half_exp < -10) {
                    // Too small to be represented -> +/-0
                } else {
                    mantissa |= 0x800000u; // restore implicit leading 1
                    uint32_t shift     = static_cast<uint32_t>(14 - half_exp);
                    uint32_t half_mant = mantissa >> shift;
                    uint32_t rem       = mantissa & ((1u << shift) - 1u);
                    uint32_t halfway   = 1u << (shift - 1);
                    // round to nearest, ties to even
                    if (rem > halfway || (rem == halfway && (half_mant & 1u))) {
                        ++half_mant;
                    }
                    fltInt16 |= static_cast<uint16_t>(half_mant);
                }
            } else {
                // Normal half; round to nearest, ties to even
                uint32_t half_mant = mantissa >> 13;
                uint32_t rem       = mantissa & 0x1fffu;
                if (rem > 0x1000u || (rem == 0x1000u && (half_mant & 1u))) {
                    ++half_mant;
                    if (half_mant == 0x400u) {
                        half_mant = 0;
                        ++half_exp; // mantissa overflow -> bump exponent
                    }
                }
                if (half_exp >= 0x1f) {
                    fltInt16 |= 0x7c00u; // rounding overflowed to Inf
                } else {
                    fltInt16 |= static_cast<uint16_t>((static_cast<uint32_t>(half_exp) << 10) | half_mant);
                }
            }
        }

        bits = fltInt16;
    }
    operator float() const {
        uint16_t bits_copy = bits;
        // NO
        return 0.0f;
    }
};
static_assert(sizeof(gpu_half) == 2, "gpu_half must be 2 bytes");

typedef shared_device_buffer_typed<gpu_half> gpu_mem_16f;
}

void run(int argc, char** argv)
{
    gpu::Device device = gpu::chooseGPUDevice(gpu::selectAllDevices(ALL_GPUS, true), argc, argv);

    gpu::Context context = activateContext(device, gpu::Context::TypeVulkan);
    // OpenCL - рекомендуется как вариант по умолчанию, можно выполнять на CPU, есть printf, есть аналог valgrind/cuda-memcheck - https://github.com/jrprice/Oclgrind
    // CUDA   - рекомендуется если у вас NVIDIA видеокарта, есть printf, т.к. в таком случае вы сможете пользоваться профилировщиком (nsight-compute) и санитайзером (compute-sanitizer, это бывший cuda-memcheck)
    // Vulkan - не рекомендуется, т.к. писать код (compute shaders) на шейдерном языке GLSL на мой взгляд менее приятно чем в случае OpenCL/CUDA
    //          если же вас это не останавливает - профилировщик (nsight-systems) при запуске на NVIDIA тоже работает (хоть и менее мощный чем nsight-compute)
    //          кроме того есть debugPrintfEXT(...) для вывода в консоль с видеокарты
    //          кроме того используемая библиотека поддерживает rassert-проверки (своеобразные инварианты с уникальным числом) на видеокарте для Vulkan

    ocl::KernelSource ocl_matrix03MultiplyNaive(ocl::getMatrix03MultiplyNaive());
    ocl::KernelSource ocl_matrix04MultiplyViaLocalMemory(ocl::getMatrix04MultiplyViaLocalMemory());

    avk2::KernelSource vk_matrix03MultiplyNaive(avk2::getMatrix03MultiplyNaive());
    avk2::KernelSource vk_matrix04MultiplyViaLocalMemory(avk2::getMatrix04MultiplyViaLocalMemory());
    avk2::KernelSource vk_matrix05MultiplyCooperativeMatrix(avk2::getMatrix05MultiplyCooperativeMatrix());

    unsigned int ksize = 128;
    unsigned int w = ksize * 32;
    unsigned int k = ksize * 8;
    unsigned int h = ksize * 16;
    std::cout << "C = A x B, matrices size: C (rows=H=" << h << " x cols=W=" << w << ")"
              << " = A (rows=H=" << h << " x cols=K=" << k << ") x B (rows=K=" << k << " x cols=W=" << w << ")" << std::endl;
    std::cout << "matrices data size: A - " << sizeof(float) * h * k / 1024 / 1024 << " MB, B - " << sizeof(float) * k * w / 1024 / 1024 << " MB, C - " << sizeof(float) * k * w / 1024 / 1024 << " MB" << std::endl;

    std::vector<float> input_a_cpu(h * k, 0);  // rows=H x cols=K
    std::vector<float> input_b_cpu(k * w, 0);  // rows=K x cols=W
    std::vector<gpu::gpu_half> input_a16_cpu(h * k, 0);  // rows=H x cols=K
    std::vector<gpu::gpu_half> input_b16_cpu(k * w, 0);  // rows=K x cols=W
    std::vector<float> output_c_cpu(h * w, 0); // rows=H x cols=W
    std::vector<float> output_c_gpu(h * w, 0); // rows=H x cols=W
    FastRandom r;
    for (size_t i = 0; i < input_a_cpu.size(); ++i) {
        float val = r.nextf();
        input_a_cpu[i] = val;
        input_a16_cpu[i] = val;
    }
    for (size_t i = 0; i < input_b_cpu.size(); ++i) {
        float val = r.nextf();
        input_b_cpu[i] = val;
        input_b16_cpu[i] = val;
    }

    // Аллоцируем буферы в VRAM
    gpu::gpu_mem_32f matrix_a_gpu(h * k); // rows=H x cols=K
    gpu::gpu_mem_32f matrix_b_gpu(k * w); // rows=K x cols=W
    gpu::gpu_mem_16f matrix_a16_gpu(h * k); // rows=H x cols=K
    gpu::gpu_mem_16f matrix_b16_gpu(k * w); // rows=K x cols=W
    gpu::gpu_mem_32f matrix_c_gpu(h * w); // rows=H x cols=W

    // Прогружаем входные данные по PCI-E шине: CPU RAM -> GPU VRAM
    matrix_a_gpu.writeN(input_a_cpu.data(), input_a_cpu.size());
    matrix_b_gpu.writeN(input_b_cpu.data(), input_b_cpu.size());
    matrix_a16_gpu.writeN(input_a16_cpu.data(), input_a16_cpu.size());
    matrix_b16_gpu.writeN(input_b16_cpu.data(), input_b16_cpu.size());

    std::vector<std::string> algorithm_names = {
        "CPU with OpenMP",
        "01 naive",
        "02 using local memory",
    };

    const std::string coop_matrix_algorithm_name = "03 using cooperative matrix [+Prestige Points]";

    int coop_matrix_size = 0;
    bool I_Want_Super_Puper_Prestige_Points = true;
    if (I_Want_Super_Puper_Prestige_Points) {
        if (context.type() == gpu::Context::TypeCUDA) {
            algorithm_names.push_back("03 using WMMA (Tensor Cores) [+Prestige Points]");
        }
        if (context.type() == gpu::Context::TypeVulkan) {
            const avk2::Device &vk_device = context.vk()->device();
            if (vk_device.supportsExtension("VK_KHR_cooperative_matrix")) {
                assert((w % COOPMAT_TILE_W == 0) && (h % COOPMAT_TILE_H == 0) && (k % COOPMAT_TILE_K == 0));

                for (const auto [dt1, dt2, m, n, k] : vk_device.supportedCooperativeMatrixSizes()) {
                    if (dt1 != DataType16f || dt2 != DataType32f) continue;
                    if (m != n || n != k) continue;
                    if (m == 0 || m > COOPMAT_TILE_K || COOPMAT_TILE_K % m != 0) continue;
                    if (COOPMAT_TILE_W % (COOPMAT_WG_W * m) != 0 || COOPMAT_TILE_H % (COOPMAT_WG_H * m) != 0) continue;

                    if (m > coop_matrix_size) {
                        coop_matrix_size = m;
                    }
                }

                if (coop_matrix_size > 0) {
                    avk2::SpecializationConstants coop_matrix_spec_constants;
                    coop_matrix_spec_constants.add(0, static_cast<uint32_t>(coop_matrix_size));
                    vk_matrix05MultiplyCooperativeMatrix.setSpecializationConstants(coop_matrix_spec_constants);
                    algorithm_names.push_back(coop_matrix_algorithm_name);
                }
            }
        }
    }

    for (size_t algorithm_index = 0; algorithm_index < algorithm_names.size(); ++algorithm_index) {
        const std::string& algorithm = algorithm_names[algorithm_index];
        std::cout << "______________________________________________________" << std::endl;
        std::cout << "Evaluating algorithm #" << (algorithm_index + 1) << "/" << algorithm_names.size() << ": " << algorithm << std::endl;

        // Обнуляем выходной буфер, чтобы результат предыдущего алгоритма не мог замаскировать ошибку текущего
        matrix_c_gpu.fill(0.0f);

        // Запускаем алгоритм (несколько раз и с замером времени выполнения)
        std::vector<double> times;
        int iters_count = (algorithm == "CPU with OpenMP") ? 1 : 10; // CPU is too slow
        for (int iter = 0; iter < iters_count; ++iter) {
            timer t;

            if (algorithm == "CPU with OpenMP") {
                cpu::multiply(input_a_cpu, input_b_cpu, output_c_cpu, w, h, k, true);
            } else {
                // _______________________________OpenCL_____________________________________________
                if (context.type() == gpu::Context::TypeOpenCL) {
                    if (algorithm == "01 naive") {
                        // TODO обязательно замените размер рабочей группы 1x1 на больший
                        ocl_matrix03MultiplyNaive.exec(gpu::WorkSize(1, 1, w, h), matrix_a_gpu, matrix_b_gpu, matrix_c_gpu, w, h, k);
                    } else if (algorithm == "02 using local memory") {
                        ocl_matrix04MultiplyViaLocalMemory.exec(gpu::WorkSize(1, 1, w, h), matrix_a_gpu, matrix_b_gpu, matrix_c_gpu, w, h, k);
                    } else {
                        rassert(false, 7652345234321, algorithm, algorithm_index);
                    }
                    // _______________________________CUDA___________________________________________
                } else if (context.type() == gpu::Context::TypeCUDA) {
                    if (algorithm == "01 naive") {
                        // TODO обязательно замените размер рабочей группы 1x1 на больший
                        cuda::matrix_multiply_naive(gpu::WorkSize(1, 1, w, h), matrix_a_gpu, matrix_b_gpu, matrix_c_gpu, w, h, k);
                    } else if (algorithm == "02 using local memory") {
                        cuda::matrix_multiply_via_local_memory(gpu::WorkSize(1, 1, w, h), matrix_a_gpu, matrix_b_gpu, matrix_c_gpu, w, h, k);
                    } else if (algorithm == "03 using WMMA (Tensor Cores) [+Prestige Points]") {
                        cuda::matrix_multiply_wmma(gpu::WorkSize(1, 1, w, h * 2 / 16), matrix_a_gpu, matrix_b_gpu, matrix_c_gpu, w, h, k);
                    } else {
                        rassert(false, 652345234321, algorithm, algorithm_index);
                    }
                    // _______________________________Vulkan_________________________________________
                } else if (context.type() == gpu::Context::TypeVulkan) {
                    struct {
                        unsigned int w;
                        unsigned int h;
                        unsigned int k;
                    } params = {w, h, k};
                    if (algorithm == "01 naive") {
                       vk_matrix03MultiplyNaive.exec(params, gpu::WorkSize(GROUP_SIZE_X, GROUP_SIZE_Y, w, h), matrix_a_gpu, matrix_b_gpu, matrix_c_gpu);
                    } else if (algorithm == "02 using local memory") {
                       vk_matrix04MultiplyViaLocalMemory.exec(params, gpu::WorkSize(GROUP_SIZE_X, GROUP_SIZE_Y, w, h), matrix_a_gpu, matrix_b_gpu, matrix_c_gpu);
                    } else if (algorithm == coop_matrix_algorithm_name) {
                        size_t tiles_w = ((size_t)(w) + COOPMAT_TILE_W - 1) / COOPMAT_TILE_W;
                        size_t tiles_h = ((size_t)(h) + COOPMAT_TILE_H - 1) / COOPMAT_TILE_H;
                        vk_matrix05MultiplyCooperativeMatrix.exec(params, gpu::WorkSize(COOP_GROUP_SIZE_X, COOP_GROUP_SIZE_Y, tiles_w * COOP_GROUP_SIZE_X, tiles_h * COOP_GROUP_SIZE_Y), matrix_a16_gpu, matrix_b16_gpu, matrix_c_gpu);
                    } else {
                        rassert(false, 7652345234321, algorithm, algorithm_index);
                    }
                } else {
                    rassert(false, 546345243, context.type());
                }
            }

            times.push_back(t.elapsed());
        }
        std::cout << "algorithm times (in seconds) - " << stats::valuesStatsLine(times) << std::endl;

        // Вычисляем достигнутую эффективную пропускную способность алгоритма
        double total_ops = 1.0 * h * w * (k + k - 1); // общее число сложений и умножений
        double gflops = 1000*1000*1000;
        std::cout << "algorithm GFlops: " << total_ops / gflops / stats::median(times) << " GFlops" << std::endl;
        int sizeof_elem = (algorithm == coop_matrix_algorithm_name ? sizeof(float) / 2 : sizeof(float));
        std::cout << "algorithm effective memory bandwidth: " << 1.0 * (h * k + k * w + h * w) * sizeof_elem / 1024 / 1024 / 1024 / stats::median(times) << " GB/s" << std::endl;

        // Сверяем результат
        if (algorithm != "CPU with OpenMP") {
            std::vector<float> results = matrix_c_gpu.readVector();
            std::vector<float> relative_errors;
            for (size_t j = 0; j < h; ++j) {
                for (size_t i = 0; i < w; ++i) {
                    float gpu_value = results[j * w + i];
                    float cpu_value = output_c_cpu[j * w + i];
                    float error = std::abs(gpu_value - cpu_value);
                    float relative_error = error / std::abs(cpu_value);
                    relative_errors.push_back(relative_error);
                }
            }
            std::cout << "relative differences with CPU: " << stats::valuesStatsLine(relative_errors) << std::endl;
            float median_relative_error = stats::median(relative_errors);
            float perc99_relative_error = stats::percentile(relative_errors, 99);
            std::cout << "median relative difference with CPU: " << median_relative_error << std::endl;
            std::cout << "99% percentile relative difference with CPU: " << perc99_relative_error << std::endl;
            rassert(median_relative_error < 1e-3f, 15321452412431, median_relative_error);
            rassert(perc99_relative_error < 1e-1f, 54623452334232, perc99_relative_error);
        }
    }
}

int main(int argc, char** argv)
{
    int exit_code = 0;
    try {
        run(argc, argv);
    } catch (std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        if (e.what() == DEVICE_NOT_SUPPORT_API) {
            // Возвращаем exit code = 0 чтобы на CI не было красного крестика о неуспешном запуске из-за выбора CUDA API (его нет на процессоре - т.е. в случае CI на GitHub Actions)
            exit_code = 0;
        } else if (e.what() == CODE_IS_NOT_IMPLEMENTED) {
            // Возвращаем exit code = 0 чтобы на CI не было красного крестика о неуспешном запуске из-за того что задание еще не выполнено
            exit_code = 0;
        } else {
            // Выставляем ненулевой exit code, чтобы сообщить, что случилась ошибка
            exit_code = 1;
        }
    }

    // we need to gracefully clear Vulkan context before it is too late (otherwise we encounter segfault on some systems)
    avk2::InstanceContext::clearGlobalInstanceContext();

    return exit_code;
}
