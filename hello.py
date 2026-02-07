import rust_cv_core

def main():
    print("Hello from Python!")
    message = rust_cv_core.hello_world()
    print(message)
    
    result = rust_cv_core.sum_as_string(5, 7)
    print(f"5 + 7 = {result}")

if __name__ == "__main__":
    main()
