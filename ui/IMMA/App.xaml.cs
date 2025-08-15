namespace IMMA
{
    public partial class App : Application
    {
        public App()
        {
            InitializeComponent();
        }

        protected override Window CreateWindow(IActivationState? activationState)
        {
            var window = new Window(new AppShell());
            window.Width = 800;
            window.Height = 920;
            return window;
        }
    }
}