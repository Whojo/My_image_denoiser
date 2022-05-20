project('FMR', 'cpp', default_options : ['cpp_std=c++20'])

include = include_directories('include')

eigen = dependency('eigen3')
gtest = dependency('gtest', main : true, required : false)

subdir('src')
subdir('test')


fft = static_library('fft', dependencies: eigen, sources: sources_fft, include_directories: include)
fft_test = executable('fft_test', sources: fft_test, dependencies: [gtest, eigen],
  link_with: fft, include_directories: include)

test('fft test', fft_test)

executable('fmr', dependencies: eigen, sources: sources_fmr, include_directories: include, link_with: fft)
