ASYNC_BEGIN

class shared_signal
{
public:
    shared_signal(bool init_val = false) : sgn(init_val) {}
    shared_signal(shared_signal &src) : sgn(src.get_sgn()) {}
    void operator=(bool val) { set_sgn(val); }
    void operator=(shared_signal &src) { set_sgn(src.get_sgn()); }
    operator bool() { return get_sgn(); }
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

class shared_counter
{
    public:
    shared_counter(uint64_t init_val = 0) : cnt(init_val) {}
    shared_counter(shared_counter &src) : cnt(src.get_cnt()) {}
    void operator=(uint64_t val) { set_cnt(val); }
    void operator=(shared_counter &src) { set_cnt(src.get_cnt()); }
    shared_counter &operator++() { increment(); return *this; }
    shared_counter operator++(int) { auto temp = *this; increment(); return temp; }
    shared_counter &operator--() { decrement(); return *this; }
    shared_counter operator--(int) { auto temp = *this; decrement(); return temp; }
    operator uint64_t() { return get_cnt(); }
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

class async_batch
{
private:
    bagrt::net_queue<std::thread> td_set;
    bagrt::net_queue<std::function<void()>> tsk_set;
    shared_counter tsk_cnt;
    shared_signal stop, proc;
    std::mutex td_mtx_tsk;
    std::condition_variable cond_tsk;
    uint64_t asyn_batch_size = ASYNC_CORE_CNT;
public:
    async_batch(uint64_t batch_size = ASYNC_CORE_CNT) : asyn_batch_size(batch_size), td_set(batch_size), tsk_set(batch_size)
    {
        for(auto i=0; i<asyn_batch_size; ++i) td_set[i] = std::thread([this](int idx){ while(true)
        {
            {
                std::unique_lock<std::mutex> lkTsk(td_mtx_tsk);
                while(!(proc || stop)) cond_tsk.wait(lkTsk);
            }
            if(stop) return;
            tsk_set[idx]();
            -- tsk_cnt;
            if(proc) proc = false;
        }}, i);
    }
    template<typename _func, typename ... _para> auto set_task(uint64_t bat_idx, _func &&func_val, _para &&...args) -> std::future<typename std::result_of<_func(_para...)>::type>
    {
        
        using ret_type = typename std::result_of<_func(_para...)>::type;
        auto p_curr_task = std::make_shared<std::packaged_task<ret_type()>>(std::bind(std::forward<_func>(func_val), std::forward<_para>(args)...));
        std::future<ret_type> res = p_curr_task->get_future();
        if(bat_idx < asyn_batch_size)
        {
            tsk_set[bat_idx] = [p_curr_task]() { (*p_curr_task)(); };
            ++ tsk_cnt;
        }
        else stop = true;
        if(tsk_cnt==asyn_batch_size)
        {
            std::unique_lock<std::mutex> lkTsk(td_mtx_tsk);
            proc = true;
            cond_tsk.notify_all();
        }
        return res;
    }
    bool is_process() { return proc; }
    uint64_t task_cnt() { return tsk_cnt; }
    ~async_batch()
    {
        stop = true;
        cond_tsk.notify_all();
        for(auto i=0; i<td_set.size(); ++i) if(td_set[i].joinable()) td_set[i].join();
    }
};

ASYNC_END