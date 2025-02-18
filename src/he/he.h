#ifndef HE_H
#define HE_H

#include <seal/seal.h>
#include <vector>
#include <cstddef> // for size_t

class CKKSPyfhel {
public:
    /**
     * @brief Constructor
     * @param poly_modulus_degree Typically 2^14 = 16384 for CKKS
     * @param scale               Typical scale = 2^30
     * @param bit_sizes          Vector of bit-lengths for the CoeffModulus
     */
     std::unique_ptr<seal::Evaluator> evaluator_;

    CKKSPyfhel(std::size_t poly_modulus_degree = 16384,
               double scale = static_cast<double>(1ULL << 30),
               const std::vector<int>& bit_sizes = {40, 30, 30, 30, 30, 30, 30, 30, 40});

    /**
     * @brief Destructor
     */
    ~CKKSPyfhel();

    /**
     * @brief Encode a double into a plaintext
     */
    seal::Plaintext encode(double value);

    /**
     * @brief Decode a plaintext into a double
     */
    double decode(const seal::Plaintext &plaintext);

    /**
     * @brief Encrypt a double, returning a ciphertext
     */
    seal::Ciphertext encrypt(double value);

    /**
     * @brief Decrypt a ciphertext, returning a double
     */
    double decrypt(const seal::Ciphertext &ciphertext);
    
    // 1D: Encode each double into a separate Plaintext
    std::vector<seal::Plaintext> encodeVector1D(const std::vector<double> &values);
    // 2D: Encode each row by calling encodeVector1D
    std::vector<std::vector<seal::Plaintext>> encodeMatrix2D(const std::vector<std::vector<double>> &mat);

    // 1D: Encrypt each double into a separate Ciphertext
    std::vector<seal::Ciphertext> encryptVector1D(const std::vector<double> &values);

    // 2D: Encrypt each element in each row
    std::vector<std::vector<seal::Ciphertext>> encryptMatrix2D(const std::vector<std::vector<double>> &mat);    

    // Decode 1D array of Plaintext -> 1D array of double
    std::vector<double> decodeVector1D(const std::vector<seal::Plaintext> &encodedVec);

    // Decode 2D array of Plaintext -> 2D array of double
    std::vector<std::vector<double>> decodeMatrix2D(const std::vector<std::vector<seal::Plaintext>> &encodedMat);

    // Decrypt 1D array of Ciphertext -> 1D array of double
    std::vector<double> decryptVector1D(const std::vector<seal::Ciphertext> &encryptedVec);

    // Decrypt 2D array of Ciphertext -> 2D array of double
    std::vector<std::vector<double>> decryptMatrix2D(const std::vector<std::vector<seal::Ciphertext>> &encryptedMat);

    /**
     * @brief Generate a new public key & secret key
     */
    void generate_keys();

    /**
     * @brief Generate relinearization keys
     */
    seal::RelinKeys generate_relin_keys();

    /**
     * @brief Get public key (serialized). Demonstrates in-memory approach.
     */
    std::string get_public_key();

    /**
     * @brief Get relin key (serialized).
     */
    seal::RelinKeys get_relin_keys() const; 

    /**
     * @brief Get relin key (serialized).
     */
    std::string get_relin_key();

    /**
     * @brief Load a public key from an in-memory string
     */
    void load_public_key(const std::string &pk_str);

    /**
     * @brief Load a relinearization key from an in-memory string
     */
    void load_relin_key(const std::string &relin_str);

    /**
     * @brief Square a ciphertext: ct^2, then relinearize & rescale.
     */
    seal::Ciphertext power2(const seal::Ciphertext &ct);

    /**
     * @brief Returns the current noise budget of a ciphertext in bits (an approximate measure).
     */
    int noise_budget(const seal::Ciphertext &ct);

private:
    // SEAL components
    seal::EncryptionParameters params_;
    std::shared_ptr<seal::SEALContext> context_;
    
    // Tools
    std::unique_ptr<seal::KeyGenerator> keygen_;
    std::unique_ptr<seal::Encryptor> encryptor_;
    std::unique_ptr<seal::Decryptor> decryptor_;
    // std::unique_ptr<seal::Evaluator> evaluator_;
    std::unique_ptr<seal::CKKSEncoder> encoder_;

    // Keys
    seal::PublicKey public_key_;
    seal::SecretKey secret_key_;
    seal::RelinKeys relin_keys_;

    // Scale used in CKKS encoding
    double scale_;
};

#endif // HE_H
