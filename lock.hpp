LOCK_BEGIN

class lock_signal
{
public:
    lock_signal() = default;
    bool get_sgn() const
    {
        std::shared_lock<std::shared_mutex> lock(shrd_mtx);
        return sgn;
    }
    void set_sgn(bool tar_sgn)
    {
        std::unique_lock<std::shared_mutex> lock(shrd_mtx);
        sgn = tar_sgn;
    }
    void set_true() { set_sgn(true); }
    void set_false() { set_sgn(false); }
private:
    mutable std::shared_mutex shrd_mtx;
    bool sgn = false;
};

class lock_counter
{
    public:
    lock_counter() = default;
    uint64_t get_cnt() const
    {
        std::shared_lock<std::shared_mutex> lock(shrd_mtx);
        return cnt;
    }
    void set_cnt(uint64_t tar_cnt = 0)
    {
        std::unique_lock<std::shared_mutex> lock(shrd_mtx);
        cnt = tar_cnt;
    }
    void increment()
    {
        std::unique_lock<std::shared_mutex> lock(shrd_mtx);
        ++ cnt;
    }
    void decrement()
    {
        std::unique_lock<std::shared_mutex> lock(shrd_mtx);
        -- cnt;
    }
private:
    mutable std::shared_mutex shrd_mtx;
    uint64_t cnt = 0;
};

LOCK_END