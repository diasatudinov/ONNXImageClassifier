require 'xcodeproj'
project = Xcodeproj::Project.new('ONNXImageClassifier.xcodeproj')
app = project.new_target(:application, 'ONNXImageClassifier', :ios, '17.0')
tests = project.new_target(:unit_test_bundle, 'ONNXImageClassifierTests', :ios, '17.0')
tests.add_dependency(app)
[app, tests].each do |target|
  target.build_configurations.each do |c|
    c.build_settings['SWIFT_VERSION'] = '6.0'
    c.build_settings['GENERATE_INFOPLIST_FILE'] = 'YES'
    c.build_settings['TARGETED_DEVICE_FAMILY'] = '1,2'
    c.build_settings['PRODUCT_BUNDLE_IDENTIFIER'] = "com.portfolio.#{target.name}"
    c.build_settings['CODE_SIGN_STYLE'] = 'Automatic'
    c.build_settings['SWIFT_STRICT_CONCURRENCY'] = 'complete'
  end
end
app.build_configurations.each do |c|
  c.build_settings['INFOPLIST_KEY_UILaunchScreen_Generation'] = 'YES'
  c.build_settings['INFOPLIST_KEY_UIApplicationSceneManifest_Generation'] = 'YES'
  c.build_settings['INFOPLIST_KEY_CFBundleDisplayName'] = 'Image Lens'
end
tests.build_configurations.each do |c|
  c.build_settings['TEST_HOST'] = '$(BUILT_PRODUCTS_DIR)/ONNXImageClassifier.app/ONNXImageClassifier'
  c.build_settings['BUNDLE_LOADER'] = '$(TEST_HOST)'
end
Dir['ONNXImageClassifier/**/*.swift'].sort.each { |path| app.add_file_references([project.main_group.new_file(path)]) }
Dir['ONNXImageClassifier/Resources/**/*'].select { |p| File.file?(p) }.sort.each { |p| app.resources_build_phase.add_file_reference(project.main_group.new_file(p)) }
Dir['ONNXImageClassifierTests/**/*.swift'].each { |p| tests.add_file_references([project.main_group.new_file(p)]) }
Dir['ONNXImageClassifierTests/Fixtures/*'].each { |p| tests.resources_build_phase.add_file_reference(project.main_group.new_file(p)) }
project.save
scheme = Xcodeproj::XCScheme.new
scheme.add_build_target(app)
scheme.add_test_target(tests)
scheme.set_launch_target(app)
scheme.save_as(project.path, 'ONNXImageClassifier', true)
