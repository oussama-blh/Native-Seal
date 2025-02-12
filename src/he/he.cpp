#include "he.h"

#include <iostream>
#include <stdexcept>
#include <cmath>
#include <sstream>

CKKSPyfhel::CKKSPyfhel(std::size_t poly_modulus_degree,
                       double scale,
                       const std::vector<int> &bit_sizes)
    : scale_(scale)
{
    // 1. Set up encryption parameters for the CKKS scheme
    params_ = seal::EncryptionParameters(seal::scheme_type::ckks);

    // 2. Set poly_modulus_degree
    params_.set_poly_modulus_degree(poly_modulus_degree);

    // 3. Set the coefficients modulus
    //    Typically we pass a vector of bit sizes like {60, 30, 30, 30, 60}
    //    to CoeffModulus::Create(...) to get the actual moduli
    params_.set_coeff_modulus(seal::CoeffModulus::Create(poly_modulus_degree, bit_sizes));

    // 4. Create the SEALContext
    context_ = std::make_shared<seal::SEALContext>(params_);

    // 5. Create needed helpers (KeyGenerator, Encoder, Encryptor, Decryptor, Evaluator)
    keygen_    = std::make_unique<seal::KeyGenerator>(*context_);
    secret_key_ = keygen_->secret_key();
    // public_key_.clear(); // not set yet until generate_keys()

    encoder_   = std::make_unique<seal::CKKSEncoder>(*context_);
    evaluator_ = std::make_unique<seal::Evaluator>(*context_);
    // We'll allocate encryptor/decryptor only after we actually have keys:
    encryptor_ = nullptr;
    decryptor_ = nullptr;
}

CKKSPyfhel::~CKKSPyfhel()
{
    // RAII usage: no manual cleanup of unique_ptr needed.
}

void CKKSPyfhel::generate_keys()
{
    // Generate public & secret key
    keygen_->create_public_key(public_key_);
    // Re-create encryptor & decryptor based on newly generated keys
    encryptor_ = std::make_unique<seal::Encryptor>(*context_, public_key_, secret_key_);
    decryptor_ = std::make_unique<seal::Decryptor>(*context_, secret_key_);
}

seal::RelinKeys CKKSPyfhel::generate_relin_keys()
{
    // Create relinearization keys
    keygen_->create_relin_keys(relin_keys_);
    return relin_keys_;
}

seal::Plaintext CKKSPyfhel::encode(double value)
{
    // We'll encode a single double into a vector of length 1
    std::vector<double> vec{ value };

    seal::Plaintext plaintext;
    encoder_->encode(vec, scale_, plaintext);
    return plaintext;
}

double CKKSPyfhel::decode(const seal::Plaintext &plaintext)
{
    // Decode into a vector<double>
    std::vector<double> decoded;
    encoder_->decode(plaintext, decoded);

    if (decoded.empty()) {
        return 0.0;
    }
    // Return first element (like your Python decodeFrac(...)[0])
    return decoded[0];
}

seal::Ciphertext CKKSPyfhel::encrypt(double value)
{
    if (!encryptor_) {
        throw std::runtime_error("Public key not generated. Call generate_keys() first.");
    }
    // Encode
    seal::Plaintext pt = encode(value);

    // Encrypt
    seal::Ciphertext ct;
    encryptor_->encrypt(pt, ct);
    return ct;
}

double CKKSPyfhel::decrypt(const seal::Ciphertext &ciphertext)
{
    if (!decryptor_) {
        throw std::runtime_error("Secret key not generated. Call generate_keys() first.");
    }
    // Decrypt
    seal::Plaintext pt;
    decryptor_->decrypt(ciphertext, pt);

    // Decode to double
    return decode(pt);
}


// 1D: Encode each double into a separate Plaintext
std::vector<seal::Plaintext> CKKSPyfhel::encodeVector1D(const std::vector<double> &values)
{
    std::vector<seal::Plaintext> encoded(values.size());
    for (size_t i = 0; i < values.size(); i++)
    {
        encoded[i] = encode(values[i]); // uses encode(double) internally
    }
    return encoded;
}

// 2D: Encode each row by calling encodeVector1D
std::vector<std::vector<seal::Plaintext>> CKKSPyfhel::encodeMatrix2D(const std::vector<std::vector<double>> &mat)
{
    std::vector<std::vector<seal::Plaintext>> encoded(mat.size());
    for (size_t r = 0; r < mat.size(); r++)
    {
        encoded[r] = encodeVector1D(mat[r]);
    }
    return encoded;
}

// 1D: Encrypt each double into a separate Ciphertext
std::vector<seal::Ciphertext> CKKSPyfhel::encryptVector1D(const std::vector<double> &values)
{
    std::vector<seal::Ciphertext> encrypted(values.size());
    for (size_t i = 0; i < values.size(); i++)
    {
        encrypted[i] = encrypt(values[i]); // uses encrypt(double) internally
    }
    return encrypted;
}

// 2D: Encrypt each element in each row
std::vector<std::vector<seal::Ciphertext>> CKKSPyfhel::encryptMatrix2D(const std::vector<std::vector<double>> &mat)
{
    std::vector<std::vector<seal::Ciphertext>> encrypted(mat.size());
    for (size_t r = 0; r < mat.size(); r++)
    {
        encrypted[r] = encryptVector1D(mat[r]);
    }
    return encrypted;
}


/******************************************************
 * 1D Decode
 *****************************************************/
std::vector<double> CKKSPyfhel::decodeVector1D(const std::vector<seal::Plaintext> &encodedVec)
{
    std::vector<double> result(encodedVec.size());
    for (size_t i = 0; i < encodedVec.size(); i++)
    {
        // decode(...) returns a single double
        result[i] = decode(encodedVec[i]);
    }
    return result;
}

/******************************************************
 * 2D Decode
 *****************************************************/
std::vector<std::vector<double>> CKKSPyfhel::decodeMatrix2D(const std::vector<std::vector<seal::Plaintext>> &encodedMat)
{
    std::vector<std::vector<double>> result(encodedMat.size());
    for (size_t r = 0; r < encodedMat.size(); r++)
    {
        result[r] = decodeVector1D(encodedMat[r]); 
    }
    return result;
}

/******************************************************
 * 1D Decrypt
 *****************************************************/
std::vector<double> CKKSPyfhel::decryptVector1D(const std::vector<seal::Ciphertext> &encryptedVec)
{
    std::vector<double> result(encryptedVec.size());
    for (size_t i = 0; i < encryptedVec.size(); i++)
    {
        // decrypt(...) returns a single double
        result[i] = decrypt(encryptedVec[i]);
    }
    return result;
}

/******************************************************
 * 2D Decrypt
 *****************************************************/
std::vector<std::vector<double>> CKKSPyfhel::decryptMatrix2D(const std::vector<std::vector<seal::Ciphertext>> &encryptedMat)
{
    std::vector<std::vector<double>> result(encryptedMat.size());
    for (size_t r = 0; r < encryptedMat.size(); r++)
    {
        result[r] = decryptVector1D(encryptedMat[r]);
    }
    return result;
}


std::string CKKSPyfhel::get_public_key()
{
    // Serialize the public key to a string
    std::ostringstream oss;
    public_key_.save(oss);
    return oss.str();
}

std::string CKKSPyfhel::get_relin_key()
{
    // Serialize the relin key to a string
    std::ostringstream oss;
    relin_keys_.save(oss);
    return oss.str();
}


seal::RelinKeys CKKSPyfhel::get_relin_keys() const {
    if (relin_keys_.data().empty()) {
        throw std::runtime_error("Relinearization keys have not been generated. Call generate_relin_keys() first.");
    }
    return relin_keys_;
}

void CKKSPyfhel::load_public_key(const std::string &pk_str)
{
    std::istringstream iss(pk_str);
    public_key_.load(*context_, iss);

    // Re-create encryptor with newly loaded public key
    encryptor_ = std::make_unique<seal::Encryptor>(*context_, public_key_, secret_key_);
}

void CKKSPyfhel::load_relin_key(const std::string &relin_str)
{
    std::istringstream iss(relin_str);
    relin_keys_.load(*context_, iss);
}

seal::Ciphertext CKKSPyfhel::power2(const seal::Ciphertext &ct)
{
    // Equivalent to "ct * ct", then relin & rescale
    seal::Ciphertext result;
    evaluator_->square(ct, result);

    // Relinearize
    if (!relin_keys_.data().empty()) {
        evaluator_->relinearize_inplace(result, relin_keys_);
    }

    // Rescale
    evaluator_->rescale_to_next_inplace(result);
    return result;
}

int CKKSPyfhel::noise_budget(const seal::Ciphertext &ct)
{
    // Returns an approximate measure of remaining noise budget in bits
    if (!decryptor_) {
        throw std::runtime_error("Secret key not generated. Cannot query noise budget.");
    }
    return decryptor_->invariant_noise_budget(ct);
}
