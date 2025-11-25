#ifndef INCLUDED_QUEUE_BROKER
#define INCLUDED_QUEUE_BROKER

#pragma once

template <unsigned int N, typename T, unsigned int MAX_THREADS>
class BrokerQueue
{
private:

	static const unsigned int MAX_THREADS_HALF = MAX_THREADS / 2;

	typedef unsigned int Ticket;
	typedef unsigned long long int HT;

	volatile Ticket tickets[N];
	T ring_buffer[N];

	HT head_tail;
	int count;

	// use nanosleep on Turing architecture, threadfence on all others
#if __CUDA_ARCH__ < 700

	template <typename L>
	__device__ __forceinline__ L atomicLoad(L* l)
	{
		return *l;
	}

	template <typename L>
	__device__ __forceinline__ L uncachedLoad(const L* l)
	{
		return *l;
	}

	__device__ __forceinline__ void backoff()
	{
		__threadfence();
	}

	__device__ __forceinline__ void sleep()
	{
		__threadfence();
	}
#else

	template <typename L>
	__device__ __forceinline__ L atomicLoad(L* l)
	{
		return *l;
	}

	template <typename L>
	__device__ __forceinline__ L uncachedLoad(const L* l)
	{
		return *l;
	}

	__device__ __forceinline__ void backoff()
	{
		__threadfence();
	}

	__device__ __forceinline__ void sleep()
	{
		__nanosleep(1);
	}
#endif

	__device__ unsigned int* head(HT* head_tail)
	{
		return reinterpret_cast<unsigned int*>(head_tail) + 1;
	}

	__device__ unsigned int* tail(HT* head_tail)
	{
		return reinterpret_cast<unsigned int*>(head_tail);
	}

	__device__ unsigned int distance(const HT& head_tail)
	{
		return static_cast<unsigned int>((head_tail & 0xFFFFFFFFULL) - (head_tail >> 32));
	}

	__device__ void waitForTicket(const unsigned int P, const Ticket number)
	{
		while (tickets[P] != number)
		{
			backoff(); // back off
		}
	}

	__device__ bool ensureDequeue()
	{
		int Num = atomicLoad(&count);
		bool ensurance = false;
		while (!ensurance && Num > 0)
		{
			ensurance = atomicSub(&count, 1) > 0;
			if (!ensurance)
			{
				Num = atomicAdd(&count, 1) + 1;
			}
		}
		return ensurance;
	}

	__device__ bool ensureEnqueue()
	{
		int Num = atomicLoad(&count);
		bool ensurance = false;
		while (!ensurance && Num < (int)N)
		{
			ensurance = atomicAdd(&count, 1) < (int)N;
			if (!ensurance)
			{
				Num = atomicSub(&count, 1) - 1;
			}
		}
		return ensurance;
	}

	__device__ void readData(T& val)
	{
		const unsigned int Pos = atomicAdd(head(const_cast<HT*>(&head_tail)), 1);
		const unsigned int P = Pos % N;

		waitForTicket(P, 2 * (Pos / N) + 1);
		val = ring_buffer[P];
		__threadfence();
		tickets[P] = 2 * ((Pos + N) / N);
	}

	__device__ void putData(const T data)
	{
		const unsigned int Pos = atomicAdd(tail(const_cast<HT*>(&head_tail)), 1);
		const unsigned int P = Pos % N;
		const unsigned int B = 2 * (Pos / N);

		waitForTicket(P, B);
		ring_buffer[P] = data;
		__threadfence();
		tickets[P] = B + 1;
	}

public:

	__device__ void init()
	{
		const int lid = threadIdx.x + blockIdx.x * blockDim.x;

		if (lid == 0)
		{
			count = 0;
			head_tail = 0x0ULL;
		}

		for (int v = lid; v < N; v += blockDim.x * gridDim.x)
		{
			// ring_buffer[v] = T(0x0);
            ring_buffer[v] = T{};     // ✅ modern C++}
			tickets[v] = 0x0;
		}
	}

	__device__ inline bool enqueue(const T& data)
	{
		bool writeData = true;
		while (writeData && !ensureEnqueue())
		{
			const unsigned int dist = distance(uncachedLoad(const_cast<unsigned long long int*>(&head_tail)));
			if (N <= dist && dist < N + MAX_THREADS_HALF)
			{
				writeData = false;
			}
			else
			{
				sleep(); //sleep a little
			}
		}

		if (writeData)
		{
			putData(data);
		}
		return writeData;
	}

	__device__ inline void dequeue(bool& hasData, T& data)
	{
		hasData = true;
		while (hasData && !ensureDequeue())
		{
			const unsigned int dist = distance(uncachedLoad(const_cast<unsigned long long int*>(&head_tail)));
			if (N + MAX_THREADS_HALF <= dist - 1)
			{
				hasData = false;
			}
			else
			{
				sleep(); //sleep a little
			}
		}

		if (hasData)
		{
			readData(data);
		}
	}
};

#endif